import time
import functools
import pandas as pd
from typing import Callable, Dict
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class FunctionStats:
    """
    Data class to store performance statistics for a specific function.
    """
    func_name: str
    module: str
    calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = float('-inf')

    @property
    def avg_time(self) -> float:
        """Calculates the average execution time."""
        return self.total_time / self.calls if self.calls > 0 else 0.0

class Profiler:
    """
    Singleton class to manage performance profiling, analysis, and reporting.
    """
    _instance = None
    _stats: Dict[str, FunctionStats] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Profiler, cls).__new__(cls)
        return cls._instance

    @classmethod
    def profile(cls, func: Callable) -> Callable:
        """
        Decorator to measure execution time of functions and methods.

        Args:
            func (Callable): The function to profile.

        Returns:
            Callable: The wrapped function with timing logic.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{func.__module__}.{func.__qualname__}"
            if key not in cls._stats:
                cls._stats[key] = FunctionStats(func.__name__, func.__module__)

            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start_time
                stats = cls._stats[key]
                stats.calls += 1
                stats.total_time += elapsed
                stats.min_time = min(stats.min_time, elapsed)
                stats.max_time = max(stats.max_time, elapsed)
        return wrapper

    def _analyze_suggestions(self, row: pd.Series) -> str:
        """
        Generates optimization suggestions based on performance heuristics.

        Args:
            row (pd.Series): A row containing function statistics.

        Returns:
            str: A suggestion string for improvement.
        """
        avg = row['avg_time']
        calls = row['calls']
        total = row['total_time']

        suggestions = []

        if avg > 1.0:
            suggestions.append("Critical: Long execution time. Consider async processing or algorithmic optimization.")
        elif avg > 0.1:
            suggestions.append("Warning: Moderate latency. Profile inner loops.")

        if calls > 1000 and avg < 0.001:
            suggestions.append("High frequency call. Consider inlining or vectorization (NumPy).")
        
        if calls > 10 and row['std_dev'] > (avg * 0.5):
            suggestions.append("Unstable performance. Check for I/O bottlenecks or variable input sizes.")

        if total > 5.0 and calls > 1:
            suggestions.append("Heavy cumulative load. Consider caching results (functools.lru_cache).")

        return " | ".join(suggestions) if suggestions else "Performance is within nominal limits."

    def generate_report(self, output_dir: Path) -> pd.DataFrame:
        """
        Compiles collected statistics into a DataFrame, analyzes them, and saves a CSV report.

        Args:
            output_dir (Path): The directory to save the report.

        Returns:
            pd.DataFrame: The generated report containing stats and suggestions.
        """
        if not self._stats:
            return pd.DataFrame()

        data = []
        for key, stat in self._stats.items():
            data.append({
                'module': stat.module,
                'function': stat.func_name,
                'calls': stat.calls,
                'total_time': stat.total_time,
                'avg_time': stat.avg_time,
                'min_time': stat.min_time,
                'max_time': stat.max_time
            })

        df = pd.DataFrame(data)
        
        df['std_dev'] = (df['max_time'] - df['min_time']) / 4
        
        df['suggestions'] = df.apply(self._analyze_suggestions, axis=1)
        
        df = df.sort_values(by='total_time', ascending=False)

        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "performance_profile.csv"
        df.to_csv(report_path, index=False)
        
        return df

profiler = Profiler()
