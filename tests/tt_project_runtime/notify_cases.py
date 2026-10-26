# Notification cases: the web app's browser notifications and the optional Slack channel's state and
# test. Run from tests/test_tt_project_runtime.py (see _include_cases there): helpers, fixtures and
# imports come from that file, and pytest collects these tests under it.
from __future__ import annotations


def _slack_api(fail_method=None, error="invalid_auth"):
    """A stand-in for the Slack Web API at the urlopen level: every call answers ok, except
    `fail_method`, which answers ok=false with `error` ('net': the network is down)."""
    calls = []

    def urlopen(req, timeout=None):
        import io
        import urllib.parse
        method = req.full_url.rsplit("/", 1)[-1]
        params = {k: v[0] for k, v in urllib.parse.parse_qs(req.data.decode()).items()}
        calls.append((method, params))
        if method == fail_method:
            if error == "net":
                raise urllib.error.URLError("connection refused")
            return io.BytesIO(json.dumps({"ok": False, "error": error}).encode())
        body = {"auth.test": {"ok": True, "bot_id": "B1"},
                "users.lookupByEmail": {"ok": True, "user": {"id": "U7"}},
                "conversations.open": {"ok": True, "channel": {"id": "D1"}},
                "chat.postMessage": {"ok": True, "ts": "1.0"}}.get(method, {"ok": True})
        return io.BytesIO(json.dumps(body).encode())
    urlopen.calls = calls
    return urlopen


def test_slack_status_names_each_state_and_says_slack_is_optional(env, monkeypatch):
    import ttp.project
    from ttp import slack
    secrets = {}
    monkeypatch.setattr(ttp.project, "load_secrets", lambda: secrets)
    off = {"notify": {"slack": False}}
    on = {"notify": {"slack": True, "slack_min_severity": "normal"}}
    got = slack.status(off)
    assert got["state"] == "not_configured" and "optional" in got["line"] and "ttp secret slack" in got["line"]
    assert slack.status(on)["state"] == "no_token" and "ttp secret slack" in slack.status(on)["line"]
    secrets["slack"] = {"bot_token": "test-bot-token", "user_id": "U1"}
    got = slack.status(off)
    assert got["state"] == "off" and "notify.slack true" in got["line"]
    got = slack.status(on)
    assert got == {"state": "on", "line": got["line"]} and "normal severity" in got["line"]
    got = slack.status(on, {"ts": 5.0, "error": "chat.postMessage: token_revoked"})
    assert got["state"] == "error" and got["since"] == 5.0 and got["line"].endswith("token_revoked")


def test_slack_send_test_reports_success_or_the_exact_error(env, monkeypatch):
    import ttp.project
    import ttp.slack
    from ttp import slack
    secrets = {}
    monkeypatch.setattr(ttp.project, "load_secrets", lambda: secrets)
    monkeypatch.setattr(ttp.slack.time, "sleep", lambda s: None)
    ok, text = slack.send_test("demo")
    assert not ok and "not configured" in text and "ttp secret slack" in text
    secrets["slack"] = {"bot_token": "test-bot-token", "user_email": "user@example.com"}
    api = _slack_api()
    monkeypatch.setattr(ttp.slack.urllib.request, "urlopen", api)
    ok, text = slack.send_test("demo")
    assert ok and "U7" in text, text
    post = [p for m, p in api.calls if m == "chat.postMessage"]
    assert len(post) == 1 and post[0]["channel"] == "D1" and post[0]["text"].startswith("[demo] Test message")
    for method, error, want in [("auth.test", "invalid_auth", "Slack test failed: auth.test: invalid_auth"),
                                ("users.lookupByEmail", "users_not_found", "users.lookupByEmail: users_not_found"),
                                ("chat.postMessage", "not_in_channel", "chat.postMessage: not_in_channel"),
                                ("auth.test", "net", "could not reach Slack")]:
        monkeypatch.setattr(ttp.slack.urllib.request, "urlopen", _slack_api(method, error))
        ok, text = slack.send_test("demo")
        assert not ok and want in text, (method, text)
    # No user to send to: the reason, not a traceback.
    secrets["slack"] = {"bot_token": "test-bot-token"}
    monkeypatch.setattr(ttp.slack.urllib.request, "urlopen", _slack_api())
    ok, text = slack.send_test("demo")
    assert not ok and "no Slack user id or email configured" in text


def test_notifier_test_slack_prints_the_outcome_and_exits_1_on_failure(env, monkeypatch, capsys):
    from ttp import cli, notifier, slack
    shown = []
    monkeypatch.setattr(notifier, "show", lambda *a, **kw: (shown.append(a), {"backend": "notify-send", "ok": True,
                        "confirmed": True, "error": "", "ts": 0.0})[1])   # show() returns its delivery result
    monkeypatch.setattr(notifier, "ensure_app", lambda: (None, ""))
    monkeypatch.setattr(slack, "send_test", lambda project="tt-project": (False, "Slack test failed: auth.test: invalid_auth"))
    with pytest.raises(SystemExit) as e:
        cli.main(["notifier", "test", "--slack"])
    assert e.value.code == 1 and "invalid_auth" in capsys.readouterr().out and shown == []
    monkeypatch.setattr(slack, "send_test", lambda project="tt-project": (True, "sent a test message to your Slack DM (user U1)"))
    cli.main(["notifier", "test", "--slack"])
    assert "sent a test message" in capsys.readouterr().out and shown == []
    cli.main(["notifier", "test"])                        # without --slack: the desktop test, as before
    assert len(shown) == 1


def test_a_failed_slack_send_shows_until_the_next_good_one(env, monkeypatch):
    import ttp.project
    from ttp import web
    failing = {"on": True}
    p, d = slack_daemon(env, monkeypatch, FakeSlackHTTP(lambda text: "token_revoked" if failing["on"] else None))
    monkeypatch.setattr(ttp.project, "load_secrets", lambda: {"slack": {"bot_token": "test-bot-token", "user_id": "U1"}})
    p.set_config("notify.slack", True)
    assert web.state_payload(p, p.db)["notify"]["slack"]["state"] == "on"
    p.db.post("in", "hi", chat="slack", channel="slack", kind="user", ref="9.0")
    p.db.post("out", "reply", chat="slack", ref="9.0")
    d.deliver_outbound()
    got = web.state_payload(p, p.db)["notify"]["slack"]
    assert got["state"] == "error" and "token_revoked" in got["line"] and got["since"], got
    failing["on"] = False
    d.deliver_outbound()
    assert web.state_payload(p, p.db)["notify"]["slack"]["state"] == "on"


@needs_sockets
def test_web_notify_test_endpoint_returns_the_notification_and_the_slack_outcome(env, monkeypatch):
    p = make(env)
    from ttp import slack, web
    sent = []
    monkeypatch.setattr(slack, "send_test", lambda project: sent.append(project) or (False, "Slack test failed: auth.test: invalid_auth"))
    port = _start_web(p)

    def post(body, token=None):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/notify/test", method="POST", data=json.dumps(body).encode(),
                                     headers={"X-TTP-Token": token or web.token(p), "Content-Type": "application/json"})
        for _ in range(50):
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())
            except OSError:
                time.sleep(0.1)
        raise AssertionError("the web app did not start")
    code, r = post({})
    assert code == 200 and r["title"] == f"{p.name}: test" and r["body"] and "slack" not in r and sent == []
    code, r = post({"slack": True})
    assert code == 200 and r["slack"] == {"ok": False, "text": "Slack test failed: auth.test: invalid_auth"} and sent == [p.name]
    assert post({}, token="wrong")[0] == 401


def test_web_app_notification_state_permission_and_announce(env):
    """The browser's state is told apart (on, ask, blocked, insecure address, unsupported), permission
    is asked only from a click, and each new ask, problem or task ready for review notifies once,
    the first one after a quiet page load too, also from a hidden tab, and stays until closed."""
    import shutil
    import subprocess
    web = RUNTIME / "ttp" / "web"
    js, html = (web / "app.js").read_text(), (web / "index.html").read_text()
    for el in ("notifycard", "notifystate", "notifyask", "notifytest", "notifyresult", "slackline", "notify"):
        assert f'id="{el}"' in html, el
    # A permission request no click started is ignored by browsers: only the buttons' handlers ask.
    assert js.count("requestPermission") == 1 and js.count("allowNotify") == 3
    assert "$(\"#notifyask\").onclick = allowNotify" in js and "return allowNotify()" in js
    # A hidden tab keeps looking by elapsed time (30 s), and looks at once when shown.
    assert "Date.now() - lastLook >= 30000" in js and "ticks" not in js
    assert 'addEventListener("visibilitychange"' in js
    node = shutil.which("node")
    if not node:
        return
    block = js[js.index("// Browser notifications while this page"):js.index('document.addEventListener("visibilitychange"')]
    stub = """
const els = {}; const $ = (s) => els[s] || (els[s] = { hidden: false, html: "", onclick: null, scrollIntoView() {} });
const text = (el, s) => { el.html = s; }, put = text, codes = (s) => s, at = (t) => String(t), api = async () => ({});
const store = {}; const sessionStorage = { getItem: (k) => k in store ? store[k] : null, setItem: (k, v) => { store[k] = String(v); } };
const shown = [], sticky = []; class Notification { constructor(t, o) { shown.push(t + " | " + o.body); sticky.push(o.requireInteraction); } }
Notification.permission = "granted";
const document = { hidden: true, querySelector: () => ({ click() {} }) };
const window = { isSecureContext: true, Notification, focus() {} };
"""
    run = """
const out = {};
out.states = [notifyState(window), notifyState({ isSecureContext: true, Notification: { permission: "denied" } }),
  notifyState({ isSecureContext: true, Notification: { permission: "default" } }),
  notifyState({ isSecureContext: false, Notification: { permission: "denied" } }), notifyState({ isSecureContext: true })];
const st = (attention, tasks) => ({ project: { name: "demo" }, attention, tasks });
announce(st([], [{ id: 4, status: "running", title: "t4" }]));                        // first look: quiet page
out.first = shown.length;
announce(st([{ id: 10, kind: "alert", text: "disk low" }], [{ id: 4, status: "review", title: "t4" }]));
announce(st([{ id: 10, kind: "alert", text: "disk low" }, { id: 11, kind: "ask", text: "merge?" }], [{ id: 4, status: "review", title: "t4" }]));
out.shown = shown.slice(); out.sticky = sticky.slice(); out.unseen = unseen;
Notification.permission = "denied"; notifyCard(); out.blocked = [els["#notifystate"].html, els["#notify"].hidden, els["#notifyask"].hidden];
Notification.permission = "default"; notifyCard(); out.ask = [els["#notify"].hidden, els["#notifyask"].hidden];
console.log(JSON.stringify(out));
"""
    r = subprocess.run([node, "-e", stub + block + run], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["states"] == ["on", "blocked", "ask", "insecure", "unsupported"]
    assert out["first"] == 0
    assert out["shown"] == ["demo: problem | disk low", "demo: ready for review | #4 t4", "demo: needs you | merge?"]
    assert out["unseen"] == 3
    assert out["sticky"] == [True, True, True]          # each stays until the user closes it
    assert "blocked by this browser" in out["blocked"][0] and "address bar" in out["blocked"][0]
    assert out["blocked"][1:] == [False, True] and out["ask"] == [False, False]


def test_a_desktop_preview_of_a_long_message_says_where_the_full_text_is():
    from ttp import notifier
    assert notifier.preview("short alert") == "short alert"
    long = notifier.preview("word " * 200)
    assert len(long) <= notifier.PREVIEW and long.endswith("(full text in the chat and web)")
    js = (RUNTIME / "ttp" / "web" / "app.js").read_text()
    assert 'preview(body, 240, "full text in the web chat")' in js and ".slice(0, 240)" not in js
