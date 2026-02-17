"""
Stub Builder — generates code skeleton strings from Tree-sitter metadata.

Instead of sending raw file content to the LLM (which wastes tokens on
function bodies, loops, and local variables), this module reconstructs
a compact "stub" that preserves only the **public interface** of the code:
signatures, decorators, docstrings, imports, and inheritance.

LLMs reason much better when they see native language syntax rather than
a JSON dictionary describing the code.
"""

from typing import Optional


class StubBuilder:
    """
    Generates language-native code stubs from file_tree metadata.

    Supported languages: Python, JavaScript, TypeScript.
    For unsupported languages, returns a structured comment block.
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def build_file_stub(self, file_info: dict) -> str:
        """
        Build a full file stub from a file_tree entry.

        Args:
            file_info: A dict from   file_tree with keys like
                       path, language, imports, function_details,
                       class_details, etc.

        Returns:
            A string of syntactically-valid stub code.
        """
        language = file_info.get("language", "")
        path = file_info.get("path", "unknown")

        parts: list[str] = []

        # Header comment
        parts.append(self._comment(f"File: {path}", language))
        if file_info.get("imports"):
            deps = ", ".join(file_info["imports"][:15])
            parts.append(self._comment(f"Dependencies: {deps}", language))
        parts.append("")

        # Imports (as literal statements)
        imports_block = self._build_imports(file_info, language)
        if imports_block:
            parts.append(imports_block)
            parts.append("")

        # Collect method parent_classes so we skip standalone methods of classes
        class_method_names = set()
        for cls in file_info.get("class_details", []):
            for m in cls.get("method_details", []):
                if isinstance(m, dict):
                    class_method_names.add(m.get("name", ""))

        # Top-level functions (those NOT inside a class)
        for func in file_info.get("function_details", []):
            if isinstance(func, dict) and not func.get("is_method", False):
                parts.append(self.build_function_stub(func, language))
                parts.append("")

        # Classes with their methods
        for cls in file_info.get("class_details", []):
            if isinstance(cls, dict):
                parts.append(self.build_class_stub(cls, language))
                parts.append("")

        result = "\n".join(parts).rstrip()
        return result if result.strip() else self._minimal_stub(file_info, language)

    def build_function_stub(self, func: dict, language: str) -> str:
        """
        Build a stub for a single function/method.

        Args:
            func: A FunctionDetail dict.
            language: The programming language (python, javascript, typescript).

        Returns:
            A string with the function signature + docstring + ellipsis body.
        """
        if language == "python":
            return self._python_function_stub(func)
        elif language in ("javascript", "typescript", "tsx"):
            return self._js_ts_function_stub(func, language)
        else:
            return self._generic_function_stub(func, language)

    def build_class_stub(self, cls: dict, language: str) -> str:
        """
        Build a stub for a class including its method stubs.

        Args:
            cls: A ClassDetail dict (with method_details).
            language: The programming language.

        Returns:
            A string with the class declaration + method stubs.
        """
        if language == "python":
            return self._python_class_stub(cls)
        elif language in ("javascript", "typescript", "tsx"):
            return self._js_ts_class_stub(cls, language)
        else:
            return self._generic_class_stub(cls, language)

    # ------------------------------------------------------------------ #
    #  Python stubs                                                       #
    # ------------------------------------------------------------------ #

    def _python_function_stub(self, func: dict, indent: str = "") -> str:
        lines: list[str] = []

        # Decorators
        for dec in func.get("decorators", []):
            lines.append(f"{indent}{dec}")

        # Signature
        keyword = "async def" if func.get("is_async") else "def"
        name = func.get("name", "unknown")
        params = ", ".join(func.get("parameters", []))
        ret = func.get("return_type")
        sig = f"{indent}{keyword} {name}({params})"
        if ret:
            sig += f" -> {ret}"
        sig += ":"
        lines.append(sig)

        # Docstring
        docstring = func.get("docstring")
        if docstring:
            lines.append(f'{indent}    """{docstring}"""')

        # Body placeholder
        lines.append(f"{indent}    ...")

        return "\n".join(lines)

    def _python_class_stub(self, cls: dict) -> str:
        lines: list[str] = []

        # Decorators
        for dec in cls.get("decorators", []):
            lines.append(dec)

        # Class declaration
        name = cls.get("name", "Unknown")
        bases = cls.get("bases", [])
        if bases:
            lines.append(f"class {name}({', '.join(bases)}):")
        else:
            lines.append(f"class {name}:")

        # Docstring
        docstring = cls.get("docstring")
        if docstring:
            lines.append(f'    """{docstring}"""')

        # Methods
        method_details = cls.get("method_details", [])
        if method_details:
            lines.append("")
            for method in method_details:
                if isinstance(method, dict):
                    lines.append(self._python_function_stub(method, indent="    "))
                    lines.append("")
        else:
            # Fallback: just list method names
            methods = cls.get("methods", [])
            if methods:
                lines.append("")
                for m in methods:
                    lines.append(f"    def {m}(self):")
                    lines.append(f"        ...")
                    lines.append("")
            else:
                lines.append("    ...")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  JavaScript / TypeScript stubs                                      #
    # ------------------------------------------------------------------ #

    def _js_ts_function_stub(self, func: dict, language: str, indent: str = "") -> str:
        lines: list[str] = []

        # Decorators
        for dec in func.get("decorators", []):
            lines.append(f"{indent}{dec}")

        # Signature
        keyword = "async function" if func.get("is_async") else "function"
        name = func.get("name", "unknown")
        params = ", ".join(func.get("parameters", []))
        ret = func.get("return_type")

        if func.get("is_method"):
            # Methods don't use the "function" keyword
            prefix = "async " if func.get("is_async") else ""
            sig = f"{indent}{prefix}{name}({params})"
        else:
            sig = f"{indent}{keyword} {name}({params})"

        if ret:
            sig += f": {ret}"

        sig += " { ... }"
        lines.append(sig)

        return "\n".join(lines)

    def _js_ts_class_stub(self, cls: dict, language: str) -> str:
        lines: list[str] = []

        # Decorators
        for dec in cls.get("decorators", []):
            lines.append(dec)

        # Class declaration
        name = cls.get("name", "Unknown")
        bases = cls.get("bases", [])
        implements = cls.get("implements", [])

        decl = f"class {name}"
        if bases:
            decl += f" extends {', '.join(bases)}"
        if implements:
            decl += f" implements {', '.join(implements)}"
        decl += " {"
        lines.append(decl)

        # Methods
        method_details = cls.get("method_details", [])
        if method_details:
            for method in method_details:
                if isinstance(method, dict):
                    lines.append(self._js_ts_function_stub(method, language, indent="  "))
        else:
            methods = cls.get("methods", [])
            for m in methods:
                lines.append(f"  {m}() {{ ... }}")

        lines.append("}")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Generic / Fallback stubs                                           #
    # ------------------------------------------------------------------ #

    def _generic_function_stub(self, func: dict, language: str) -> str:
        lines: list[str] = []
        name = func.get("name", "unknown")
        params = ", ".join(func.get("parameters", []))
        ret = func.get("return_type", "")
        docstring = func.get("docstring", "")

        lines.append(self._comment(f"function: {name}({params})" + (f" -> {ret}" if ret else ""), language))
        if docstring:
            lines.append(self._comment(docstring, language))

        return "\n".join(lines)

    def _generic_class_stub(self, cls: dict, language: str) -> str:
        lines: list[str] = []
        name = cls.get("name", "Unknown")
        bases = cls.get("bases", [])
        docstring = cls.get("docstring", "")
        methods = cls.get("methods", [])

        header = f"class: {name}"
        if bases:
            header += f" extends {', '.join(bases)}"
        lines.append(self._comment(header, language))
        if docstring:
            lines.append(self._comment(docstring, language))
        if methods:
            lines.append(self._comment(f"methods: {', '.join(methods)}", language))

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _comment(self, text: str, language: str) -> str:
        """Create a single-line comment in the appropriate language syntax."""
        if language == "python":
            return f"# {text}"
        elif language in ("javascript", "typescript", "tsx", "java", "go", "rust", "c", "cpp", "csharp", "kotlin", "swift", "scala"):
            return f"// {text}"
        else:
            return f"# {text}"

    def _build_imports(self, file_info: dict, language: str) -> str:
        """Build import statements from file_info."""
        imports = file_info.get("imports", [])
        if not imports:
            return ""

        lines: list[str] = []
        if language == "python":
            for imp in imports[:20]:
                lines.append(f"import {imp}")
        elif language in ("javascript", "typescript", "tsx"):
            for imp in imports[:20]:
                if imp.startswith(".") or imp.startswith("/"):
                    lines.append(f"import '...' from '{imp}'")
                else:
                    lines.append(f"import {imp}")
        else:
            for imp in imports[:15]:
                lines.append(self._comment(f"import: {imp}", language))

        return "\n".join(lines)

    def _minimal_stub(self, file_info: dict, language: str) -> str:
        """Fallback stub when there are no functions or classes."""
        parts: list[str] = []
        parts.append(self._comment(f"File: {file_info.get('path', 'unknown')}", language))
        parts.append(self._comment(f"Language: {language}", language))
        parts.append(self._comment(f"Lines: {file_info.get('line_count', '?')}", language))

        functions = file_info.get("functions", [])
        classes = file_info.get("classes", [])
        if functions:
            parts.append(self._comment(f"Functions: {', '.join(functions[:10])}", language))
        if classes:
            parts.append(self._comment(f"Classes: {', '.join(classes[:10])}", language))

        return "\n".join(parts)
