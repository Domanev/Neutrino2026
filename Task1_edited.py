import numpy as np


class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5):
        if R <= 0 or r <= 0:
            raise ValueError("Радиусы R и r должны быть положительными.")
        if not isinstance(N, (int, np.integer)) or N < 1:
            raise ValueError("Число ФЭУ N должно быть положительным целым.")

        self.R = R
        self.N = N
        self.r = r
        self.a = a

        self.fmt_positions = self.generate_fmts()

        # Достаточное условие отсутствия перекрытий: расстояния >= 2r.
        diff = self.fmt_positions[:, None, :] - self.fmt_positions[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(distances, np.inf)
        if np.min(distances) < 2 * self.r:
            raise ValueError(
                "Расстояние между центрами ФЭУ меньше 2r. Уменьшите r или N."
            )

    def generate_fmts(self):
        """Разместим N центров ФЭУ по сфере радиуса R методом Фибоначчи."""
        i = np.arange(self.N)

        cos_theta = 1 - 2 * (i + 0.5) / self.N
        sin_theta = np.sqrt(1 - cos_theta**2)

        golden_angle = np.pi * (3 - np.sqrt(5))
        phi = i * golden_angle

        x = self.R * sin_theta * np.cos(phi)
        y = self.R * sin_theta * np.sin(phi)
        z = self.R * cos_theta

        return np.column_stack([x, y, z])

    def find_fmt(self, point):
        """
        Проверим, попадает ли точка на поверхности детектора на ФЭУ.
        Радиус области ФЭУ задается расстоянием по хорде до его центра.
        Возвращает номер ФЭУ (0..N-1) или -1, если мимо.
        """
        diff = self.fmt_positions - point
        dist = np.sqrt(np.sum(diff**2, axis=1))

        best_fmt = np.argmin(dist)
        if dist[best_fmt] <= self.r:
            return int(best_fmt)
        return -1


if __name__ == "__main__":
    from pathlib import Path
    import webbrowser

    import k3d

    det = Detector(R=2.0, N=40, r=0.1, a=0.5)
    plot = k3d.plot(name="Детектор")

    fmt_mesh = k3d.points(
        positions=det.fmt_positions.astype(np.float32),
        point_size=det.r,
        color=0x000000,
    )
    plot += fmt_mesh

    page = Path(__file__).resolve().with_name("detector.html")
    page.write_text(plot.get_snapshot(), encoding="utf-8")
    webbrowser.open(page.as_uri())
