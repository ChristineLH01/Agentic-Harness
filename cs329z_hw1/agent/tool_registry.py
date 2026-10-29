"""Problem 5 (tool_registry): the toolbox the agent can call."""

from __future__ import annotations

from jsonschema import Draft202012Validator

from cs329z_hw1.types import ToolResult, ToolSpec


class ToolRegistry:
    """Holds the tools by name and validates a call separately from running it."""

    def __init__(self, tools: list[ToolSpec]) -> None:
        self.tools = {t.name: t for t in tools}  # dicts keep registration order
        self.validators = {t.name: Draft202012Validator(t.parameters) for t in tools}

    def schemas(self) -> list[dict]:
        """What the model sees of each tool, in registration order."""
        return [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in self.tools.values()]

    def validate(self, name: str, args: dict) -> ToolResult | None:
        """The unknown_tool or invalid_args result the call would get, or None if it can run."""
        if not isinstance(name, str) or name not in self.tools:
            return ToolResult("unknown_tool", f"Unknown tool {name!r}. Available tools: {', '.join(self.tools) or 'none'}.")
        errors = sorted(self.validators[name].iter_errors(args), key=lambda e: list(e.path))
        if errors:
            problems = "; ".join(f"{'.'.join(map(str, e.path)) or 'arguments'}: {e.message}" for e in errors)
            return ToolResult("invalid_args", f"Invalid arguments for {name}: {problems}")
        return None

    def execute(self, name: str, args: dict) -> ToolResult:
        """Validate, then run the tool. Never raises."""
        rejected = self.validate(name, args)
        if rejected:
            return rejected
        try:
            return ToolResult("ok", str(self.tools[name].fn(**args)))
        except Exception as exc:
            return ToolResult("tool_error", f"{name} failed: {type(exc).__name__}: {exc}")
