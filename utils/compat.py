"""Python 3.10+ compatibility shim for experta (collections.Mapping)."""


def patch_experta_collections() -> None:
    import collections

    if not hasattr(collections, "Mapping"):
        import collections.abc

        collections.Mapping = collections.abc.Mapping  # type: ignore[attr-defined]
