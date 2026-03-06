Create a GitHub issue based on the conversation context and the instruction below.

1. Get the current branch name with `git branch --show-current`
2. Create the issue using `gh issue create` with:
   - A concise title derived from the instruction
   - A body that includes:
     - The instruction expanded into a clear description
     - A line: "Branch: `<current branch name>`"
     - A line at the end: "@claude"

Instruction: $ARGUMENTS
