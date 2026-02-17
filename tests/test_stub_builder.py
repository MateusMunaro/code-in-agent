"""
Smoke test for the StubBuilder service.

Validates that stubs are generated correctly from mock file_tree data.
Run: python -m agent.tests.test_stub_builder
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.src.services.stub_builder import StubBuilder


def test_python_file_stub():
    """Test full file stub generation for Python."""
    file_info = {
        "path": "src/services/user_service.py",
        "language": "python",
        "line_count": 800,
        "imports": ["flask", "sqlalchemy", "services.email"],
        "functions": ["create_user", "validate_email"],
        "classes": ["UserService"],
        "function_details": [
            {
                "name": "validate_email",
                "start_line": 10,
                "end_line": 25,
                "parameters": ["email: str"],
                "return_type": "bool",
                "decorators": [],
                "docstring": "Validate email format.",
                "is_async": False,
                "is_method": False,
                "parent_class": None,
            },
        ],
        "class_details": [
            {
                "name": "UserService",
                "start_line": 30,
                "end_line": 800,
                "bases": ["BaseService"],
                "decorators": ["@injectable"],
                "methods": ["__init__", "create_user", "get_user"],
                "method_details": [
                    {
                        "name": "__init__",
                        "start_line": 35,
                        "end_line": 40,
                        "parameters": ["self", "repo: UserRepository", "email: EmailService"],
                        "return_type": None,
                        "decorators": [],
                        "docstring": None,
                        "is_async": False,
                        "is_method": True,
                        "parent_class": "UserService",
                    },
                    {
                        "name": "create_user",
                        "start_line": 42,
                        "end_line": 100,
                        "parameters": ["self", "payload: UserCreateDTO"],
                        "return_type": "User",
                        "decorators": ["@transactional"],
                        "docstring": "Cria um usuário e dispara email de boas-vindas.",
                        "is_async": False,
                        "is_method": True,
                        "parent_class": "UserService",
                    },
                    {
                        "name": "get_user",
                        "start_line": 102,
                        "end_line": 120,
                        "parameters": ["self", "user_id: int"],
                        "return_type": "Optional[User]",
                        "decorators": [],
                        "docstring": "Busca um usuário pelo ID.",
                        "is_async": True,
                        "is_method": True,
                        "parent_class": "UserService",
                    },
                ],
                "docstring": "Serviço responsável pela orquestração de regras de negócio de usuários.",
                "implements": [],
            },
        ],
    }

    builder = StubBuilder()
    stub = builder.build_file_stub(file_info)

    print("=" * 60)
    print("PYTHON FILE STUB (original: 800 lines)")
    print("=" * 60)
    print(stub)
    print(f"\n📊 Stub: {len(stub)} chars, {stub.count(chr(10)) + 1} lines")
    print()

    # Assertions
    assert "class UserService(BaseService):" in stub, "Class declaration missing"
    assert "@injectable" in stub, "Class decorator missing"
    assert "@transactional" in stub, "Method decorator missing"
    assert "async def get_user" in stub, "Async method missing"
    assert '"""Cria um usuário' in stub, "Docstring missing"
    assert "import flask" in stub, "Import missing"
    assert "..." in stub, "Ellipsis body placeholder missing"
    assert "def validate_email(email: str) -> bool:" in stub, "Top-level func missing"
    print("✅ All Python assertions passed!\n")


def test_typescript_file_stub():
    """Test full file stub generation for TypeScript."""
    file_info = {
        "path": "src/controllers/user.controller.ts",
        "language": "typescript",
        "line_count": 500,
        "imports": ["express", "./services/user.service", "@nestjs/common"],
        "exports": ["UserController"],
        "functions": [],
        "classes": ["UserController"],
        "function_details": [],
        "class_details": [
            {
                "name": "UserController",
                "start_line": 10,
                "end_line": 500,
                "bases": [],
                "decorators": ["@Controller('users')"],
                "methods": ["getUser", "createUser"],
                "method_details": [
                    {
                        "name": "getUser",
                        "start_line": 15,
                        "end_line": 30,
                        "parameters": ["id: string"],
                        "return_type": "Promise<User>",
                        "decorators": ["@Get(':id')"],
                        "docstring": None,
                        "is_async": True,
                        "is_method": True,
                        "parent_class": "UserController",
                    },
                    {
                        "name": "createUser",
                        "start_line": 32,
                        "end_line": 60,
                        "parameters": ["body: CreateUserDto"],
                        "return_type": "User",
                        "decorators": ["@Post()"],
                        "docstring": None,
                        "is_async": True,
                        "is_method": True,
                        "parent_class": "UserController",
                    },
                ],
                "docstring": None,
                "implements": ["IUserController"],
            },
        ],
    }

    builder = StubBuilder()
    stub = builder.build_file_stub(file_info)

    print("=" * 60)
    print("TYPESCRIPT FILE STUB (original: 500 lines)")
    print("=" * 60)
    print(stub)
    print(f"\n📊 Stub: {len(stub)} chars, {stub.count(chr(10)) + 1} lines")
    print()

    # Assertions
    assert "class UserController" in stub, "Class declaration missing"
    assert "implements IUserController" in stub, "Implements missing"
    assert "@Controller('users')" in stub, "Class decorator missing"
    assert "@Get(':id')" in stub, "Method decorator missing"
    assert "async getUser" in stub, "Async method missing"
    assert "{ ... }" in stub, "Body placeholder missing"
    print("✅ All TypeScript assertions passed!\n")


def test_function_stub_isolation():
    """Test isolated function stub generation."""
    builder = StubBuilder()

    func = {
        "name": "process_data",
        "parameters": ["data: list[dict]", "config: Config"],
        "return_type": "ProcessResult",
        "decorators": ["@cache(ttl=300)"],
        "docstring": "Processa dados em lote com configuração.",
        "is_async": True,
        "is_method": False,
    }

    stub = builder.build_function_stub(func, "python")

    print("=" * 60)
    print("ISOLATED FUNCTION STUB")
    print("=" * 60)
    print(stub)
    print()

    assert "@cache(ttl=300)" in stub
    assert "async def process_data" in stub
    assert "-> ProcessResult:" in stub
    assert "..." in stub
    print("✅ Function stub assertions passed!\n")


if __name__ == "__main__":
    print("\n🧪 Running StubBuilder Smoke Tests\n")
    test_python_file_stub()
    test_typescript_file_stub()
    test_function_stub_isolation()
    print("🎉 All smoke tests passed!")
