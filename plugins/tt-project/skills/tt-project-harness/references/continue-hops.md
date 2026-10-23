# Continue hops

A done task's only follow-up titled `continue: ...` is queued by the daemon at the parent's tier (caps under `chains`).
A hop that only runs checks, pushes or edits spec or PR text costs less at light: title it `mechanical: ...`, or give the follow-up `"tier": "light"` (lower only, never above the parent's).
Its spec is unchanged; a light hop that needs judgment is retried a tier up.

A code task that splits (its branch has changes and it proposes a `continue:` hop, or its hand-off or head commit marks the head WIP or split) gets no review of its own.
Its hop takes the branch over and starts at once: never give it a `landed:#<task>` wait on its own chain (refused), as that work lands only with the hop's.
The chain's last hop, the first that ends without a split, gets one review of the whole chain; a failed review's findings go to an open hop of that branch, unless that hop waits on a task of the chain landing (it can add nothing first): then a Fix review task takes them.
