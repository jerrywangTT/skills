# Memory tidy-up

`ttp memory <name> --forget <entry> <entry> ... [--why W]` retires several entries in one call: one MEMORY.md rewrite and one harness commit, far faster than one call per entry.
A name it cannot retire (unknown, ambiguous, standing without `--why`) is reported and the command exits non-zero; the others are still retired.
To merge overlapping entries use `ttp memory <name> "<merged>" --kind K --supersedes <entry>...` (one commit; refused whole if it would drop a standing flag, change a user rule's kind or lose the user's quoted words).
`--title T` sets the entry's title (its MEMORY.md line and file name) and `--audience workers|coordinator` who gets it, on a new entry or a merged one; a merge with no `--audience` keeps the one the replaced entries share, and `--audience default` clears it. A merge keeps the replaced entries' earliest end (expires, until, until_probe); `--permanent` drops it. A consolidation of pinned worker memory runs this way: `ttp memory <name> "<merged>" --kind restriction --title T --audience workers --supersedes <entry>...`.
Worker prompts leave out a memory restriction whose every sentence the charter's Restrictions state (the binding block has it), and do not count it against their memory budget.
`--search <words>` finds live and archived entries.
Decisions and facts no prompt has shown for `memory.archive_hidden_days` days (default 7) are archived daily, model-free.
