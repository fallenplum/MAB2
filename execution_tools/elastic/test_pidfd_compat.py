import errno
import unittest
from unittest import mock
import elastic_migrate as migration


class PidfdCompatibility(unittest.TestCase):
    def test_python_handle_api_is_preferred(self):
        with mock.patch.object(migration.os, "pidfd_open", return_value=41, create=True) as opened:
            with mock.patch.object(migration, "checked_syscall") as native:
                self.assertEqual(migration.pidfd_open(123), 41)
                opened.assert_called_once_with(123)
                native.assert_not_called()

    def test_python_signal_api_is_preferred(self):
        with mock.patch.object(migration.signal, "pidfd_send_signal", return_value=0, create=True) as send:
            with mock.patch.object(migration, "checked_syscall") as native:
                migration.pidfd_signal(41, 0)
                send.assert_called_once_with(41, 0)
                native.assert_not_called()

    def test_unsupported_platform_fails_before_any_syscall(self):
        with mock.patch.object(migration.sys, "platform", "win32"):
            with mock.patch.object(migration.ctypes, "CDLL") as library:
                with self.assertRaises(RuntimeError):
                    migration.checked_syscall(434)
                library.assert_not_called()

    def test_kernel_error_is_not_ignored(self):
        library = mock.Mock()
        library.syscall.return_value = -1
        with mock.patch.object(migration.sys, "platform", "linux"), mock.patch.object(migration.platform, "machine", return_value="x86_64"), mock.patch.object(migration.ctypes, "CDLL", return_value=library), mock.patch.object(migration.ctypes, "get_errno", return_value=errno.ENOSYS):
            with self.assertRaises(OSError) as caught:
                migration.checked_syscall(434)
            self.assertEqual(caught.exception.errno, errno.ENOSYS)

    def test_preflight_queries_only_self_and_closes_handle_on_failure(self):
        with mock.patch.object(migration, "pidfd_open", return_value=41) as opened, mock.patch.object(migration, "pidfd_signal", side_effect=OSError("unsupported")) as send, mock.patch.object(migration.os, "close") as close:
            with self.assertRaises(OSError):
                migration.pidfd_preflight()
            opened.assert_called_once_with(migration.os.getpid())
            send.assert_called_once_with(41, 0)
            close.assert_called_once_with(41)


if __name__ == "__main__":
    unittest.main()
