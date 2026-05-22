Read INFO.md in the repository root for an LLM generated project overview. Update it if we make changes or find new information or corrections we can make.

Don't make unrelated edits unless I specifically ask for it. Do not proactively change code, configs, paths, or behavior when I only ask a question, show an error, or ask for diagnosis; explain the issue and ask before patching. If I phrase a concrete desired codebase change as a question, such as "can we make X do Y" or "can we give X instead of Y", treat it as a patch request and implement it. If you find a bug, just let me know and ask if I want to patch it. Don't blindly add support fallbacks and different cases, that leaves the code long, messy, and with too many cases to track. Change the structure of the code once and that's it. Keep any written code clean, modular, readable, and concise. Avoid the project becoming a patchwork of patches, at the end of each reply you should either ask a question or be finished implementing. If you're finished implementing, the state of the codebase should be production-grade. Before editing or planning, make sure you have read the README.md. After editing, make sure you update the README.md if necessary so we continue to understand exactly what a product does and how it works. When using python locally, (like for smoke tests) use the virtual environments in the repo. 

Keep all new additions as simple as possible. Don't add more code than necessary to fix the root issue.

When we start a new problem or feature request, be deliberate:

Deconstruct. What's the actual underlying issue? Don't take my framing for granted.
Consider alternatives. From a whole-codebase view, is my proposed approach the best one, or is there something more elegant?
Propose the best plan. Briefly, so I can push back before you build.

Then follow these principles when writing code:

Match existing patterns before inventing new ones. Consistency beats individual "better" choices. Don't Repeat Yourself, modularization, and reusability!
Smallest diff that solves it. No drive-by refactors. No speculative abstraction — wait for the third occurrence before extracting.
No quiet failures. Log or raise, never swallow. Make the choice explicit.
Make side effects obvious. Pure functions where possible; when something mutates state or hits I/O, it should look like it does.
Ask when ambiguous. A clarifying question beats a wrong-interpretation PR.k
State what you didn't do. Flag skipped edge cases, assumptions, and deferred cleanups when you finish.


At the end of each successful finished code/documentation change, concisely suggest a commit message.
