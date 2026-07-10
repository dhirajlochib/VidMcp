"""Plugin/recipe marketplace — load, save, list shareable recipe packs."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson

from vidmcp.harness.recipes import RECIPES, get_recipe, list_recipes
from vidmcp.utils.logging import get_logger

log = get_logger("vidmcp.marketplace")


class RecipeMarketplace:
    def __init__(self, root: Path):
        self.root = Path(root) / "marketplace" / "recipes"
        self.root.mkdir(parents=True, exist_ok=True)
        self._installed: dict[str, dict[str, Any]] = {}
        self._load_disk()

    def _load_disk(self) -> None:
        for p in self.root.glob("*.json"):
            try:
                data = orjson.loads(p.read_bytes())
                name = data.get("name") or p.stem
                self._installed[name] = data
            except Exception:
                continue

    def list_all(self) -> list[dict[str, Any]]:
        builtin = list_recipes()
        for b in builtin:
            b["source"] = "builtin"
        community = []
        for name, data in self._installed.items():
            community.append(
                {
                    "name": name,
                    "description": data.get("description", ""),
                    "subject_prompt": data.get("subject_prompt"),
                    "source": "marketplace",
                    "author": data.get("author", "unknown"),
                    "version": data.get("version", "0.1.0"),
                }
            )
        return builtin + community

    def get(self, name: str) -> dict[str, Any]:
        if name in self._installed:
            return dict(self._installed[name])
        return get_recipe(name)

    def publish(self, recipe: dict[str, Any], *, author: str = "local") -> dict[str, Any]:
        name = recipe.get("name") or f"recipe_{uuid4().hex[:8]}"
        recipe = {**recipe, "name": name, "author": author, "version": recipe.get("version") or "0.1.0"}
        path = self.root / f"{name}.json"
        path.write_bytes(orjson.dumps(recipe, option=orjson.OPT_INDENT_2))
        self._installed[name] = recipe
        # also inject into runtime RECIPES for apply_recipe
        RECIPES[name] = recipe
        return {"ok": True, "name": name, "path": str(path)}

    def install_from_path(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        data = orjson.loads(path.read_bytes())
        return self.publish(data, author=data.get("author") or "imported")

    def export_pack(self, names: list[str], out_path: Path) -> dict[str, Any]:
        pack = {"format": "vidmcp-recipe-pack-v1", "recipes": []}
        for n in names:
            try:
                pack["recipes"].append(self.get(n))
            except Exception:
                continue
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(orjson.dumps(pack, option=orjson.OPT_INDENT_2))
        return {"ok": True, "path": str(out_path), "count": len(pack["recipes"])}

    def import_pack(self, path: Path) -> dict[str, Any]:
        data = orjson.loads(Path(path).read_bytes())
        recipes = data.get("recipes") or []
        installed = []
        for r in recipes:
            res = self.publish(r, author=r.get("author") or "pack")
            installed.append(res["name"])
        return {"ok": True, "installed": installed}
