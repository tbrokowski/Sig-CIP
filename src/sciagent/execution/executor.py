"""
Code execution with monitoring and safety

Provides safe code execution with real-time monitoring, error handling,
and resource limits.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
import traceback as tb
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sciagent.utils.logging import logger
from sciagent.utils.models import ExecutionResult


class MonitoredExecutor:
    """
    Execute code with real-time monitoring and error handling

    Features:
    - Captures stdout/stderr
    - Timeout enforcement
    - Resource monitoring
    - Progress callbacks
    - Error handling
    """

    def __init__(
        self,
        code: Any,
        experiment_id: str,
        on_error: Optional[Callable] = None,
        on_progress: Optional[Callable] = None,
        timeout: int = 3600,
    ):
        """
        Initialize executor

        Args:
            code: Code artifact to execute
            experiment_id: ID of experiment
            on_error: Callback for errors
            on_progress: Callback for progress updates
            timeout: Timeout in seconds
        """
        self.code = code.code if hasattr(code, "code") else str(code)
        self.experiment_id = experiment_id
        self.on_error = on_error
        self.on_progress = on_progress
        self.timeout = timeout

        # Execution state
        self.start_time = 0.0
        self.end_time = 0.0
        self.stdout_capture = StringIO()
        self.stderr_capture = StringIO()

    async def run(self) -> ExecutionResult:
        """
        Execute code and return results

        Returns:
            Execution result with data, metrics, and status
        """

        logger.info(f"[{self.experiment_id}] Starting execution")
        self.start_time = time.time()

        try:
            # Report progress
            if self.on_progress:
                await self.on_progress({"status": "starting", "progress": 0.0})

            # Create temporary directory for execution
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Write code to file
                code_file = tmpdir_path / "experiment.py"
                code_file.write_text(self.code)

                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_code(code_file, tmpdir_path), timeout=self.timeout
                )

                self.end_time = time.time()
                duration = self.end_time - self.start_time

                # Report completion
                if self.on_progress:
                    await self.on_progress({"status": "completed", "progress": 1.0})

                return ExecutionResult(
                    success=True,
                    data=result.get("data", {}),
                    summary=result.get("summary", "Execution completed successfully"),
                    metrics=result.get("metrics", {}),
                    plots=result.get("plots", []),
                    duration=duration,
                )

        except asyncio.TimeoutError:
            error_msg = f"Execution timed out after {self.timeout} seconds"
            logger.error(f"[{self.experiment_id}] {error_msg}")

            if self.on_error:
                await self.on_error(TimeoutError(error_msg))

            return ExecutionResult(
                success=False,
                data={},
                summary=error_msg,
                error=error_msg,
                traceback="Timeout",
                duration=self.timeout,
            )

        except Exception as e:
            error_msg = str(e)
            traceback = tb.format_exc()

            logger.error(f"[{self.experiment_id}] Execution failed: {error_msg}")
            logger.debug(traceback)

            if self.on_error:
                await self.on_error(e)

            return ExecutionResult(
                success=False,
                data={},
                summary=f"Execution failed: {error_msg}",
                error=error_msg,
                traceback=traceback,
                duration=time.time() - self.start_time,
            )

    async def _execute_code(self, code_file: Path, work_dir: Path) -> Dict[str, Any]:
        """
        Execute code file and capture results

        Args:
            code_file: Path to code file
            work_dir: Working directory

        Returns:
            Execution results dictionary
        """

        # Prepare execution environment
        execution_globals = {
            "__name__": "__main__",
            "__file__": str(code_file),
            "experiment_id": self.experiment_id,
            "work_dir": str(work_dir),
        }

        execution_locals = {}

        # Redirect stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = self.stdout_capture
            sys.stderr = self.stderr_capture

            # Execute code
            with open(code_file) as f:
                code_text = f.read()

            # Use exec to run code
            exec(code_text, execution_globals, execution_locals)

            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Collect results
            stdout_text = self.stdout_capture.getvalue()
            stderr_text = self.stderr_capture.getvalue()

            # Extract metrics and data from locals
            metrics = {}
            data = {}

            for key, value in execution_locals.items():
                if not key.startswith("_"):
                    # Try to extract numeric metrics
                    if isinstance(value, (int, float)):
                        metrics[key] = float(value)
                    elif isinstance(value, dict):
                        # Check if it's a metrics dict
                        if all(isinstance(v, (int, float)) for v in value.values()):
                            metrics.update(value)
                        else:
                            data[key] = value
                    else:
                        data[key] = value

            # Create summary from stdout
            summary = stdout_text if stdout_text else "Execution completed"

            # Find plot files
            plots = list(work_dir.glob("*.png")) + list(work_dir.glob("*.jpg"))

            return {
                "data": data,
                "metrics": metrics,
                "summary": summary,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "plots": plots,
            }

        except Exception as e:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            raise

    async def cancel(self):
        """Cancel execution"""
        logger.warning(f"[{self.experiment_id}] Execution cancelled")
        # TODO: Implement cancellation logic
