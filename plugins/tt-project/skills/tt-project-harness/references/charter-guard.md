# Charter guard

The coordinator's `charter_update` (runtime `coordinator.py`, `_reject_contradicting_append`) refuses added text that a standing Restrictions item would contradict, quoting the item and naming the forms that pass: a quoted resend, `both_hold` plus a companion edit for an exception, or `both_hold` (or the text saying it loosens no restriction) when both hold.
A companion edit keeps the quoted item word for word and only adds a clause naming, by id, an exception that stands in a temporary or dated Restrictions section, with no object or action of its own, on the sentence that exception is about.
The user's yes to that exception covers it in a later turn and it is checked against its own item only.
Removing or really loosening an item still needs the user's word or `over`.
The refusal says a user's loosening goes in as written (a quoted resend on their word, no `over` and no end they did not give), and names the temporary-exception route only when the user's own words make it temporary.
After such a loosening or removal (`quote`, or `replaces` of a Restrictions section, judged item by item), or an `over` retirement, a `restriction_tools` event (runtime `restriction_audit.py`) names the tool files still carrying the old item's terms; it never edits them.
A refusal that names such a fix raises an immediate `charter_retry` event, at most two per change.
Raise `CHARTER_GUARD_VERSION` and its pin when the guard changes.
