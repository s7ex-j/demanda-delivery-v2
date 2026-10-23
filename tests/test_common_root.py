import unittest
from scripts.figure_scripts.figuras.common import get_project_root

class TestProjectRoot(unittest.TestCase):
    def test_root_detection(self):
        root = get_project_root()
        self.assertTrue((root / ".git").exists() or (root / "pyproject.toml").exists())

    def test_root_paths_exist(self):
        """Verifica que las rutas principales derivadas del ROOT existan."""
        root = get_project_root()
        # Verificar que las rutas usadas en common.py existen
        data_path = root / "data" / "processed" / "demanda_limpia.csv"
        reports_path = root / "reports"
        fig_dir = root / "Informe" / "figuras"

        self.assertTrue(data_path.exists(), f"No existe: {data_path}")
        self.assertTrue(reports_path.exists(), f"No existe: {reports_path}")
        self.assertTrue(fig_dir.exists(), f"No existe: {fig_dir}")

if __name__ == "__main__":
    unittest.main()