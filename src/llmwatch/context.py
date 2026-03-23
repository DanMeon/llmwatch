"""ContextVar-based tag stack for nested tracking contexts."""

from contextvars import ContextVar

from llmwatch.schemas.tags import Tags

# ^ Tag stack: supports nested contexts (asyncio safe)
_tag_stack: ContextVar[list[Tags]] = ContextVar("llmwatch_tag_stack", default=[])


def push_tags(tags: Tags) -> None:
    """Push tags onto the stack."""
    stack = _tag_stack.get().copy()
    stack.append(tags)
    _tag_stack.set(stack)


def pop_tags() -> Tags:
    """Pop tags from the stack."""
    stack = _tag_stack.get().copy()
    if not stack:
        return Tags()
    tags = stack.pop()
    _tag_stack.set(stack)
    return tags


def get_current_tags() -> Tags:
    """Merge all tags in the stack and return the current active tags. Lower entries override upper ones."""
    stack = _tag_stack.get()
    if not stack:
        return Tags()
    merged = Tags()
    for tags in stack:
        merged = merged.merge(tags)
    return merged
