from backend.security import ALLOWED_MIME_TYPES, is_private_ip, mask_url, validate_external_url


class TestAllowedMimeTypes:
    def test_octet_stream_not_allowed(self):
        # Catch-all type must be rejected so unknown binaries can't bypass the check
        assert "application/octet-stream" not in ALLOWED_MIME_TYPES

    def test_common_media_allowed(self):
        assert "audio/mpeg" in ALLOWED_MIME_TYPES
        assert "video/mp4" in ALLOWED_MIME_TYPES


class TestIsPrivateIp:
    def test_loopback(self):
        assert is_private_ip("127.0.0.1")
        assert is_private_ip("::1")

    def test_rfc1918(self):
        assert is_private_ip("10.0.0.1")
        assert is_private_ip("192.168.1.1")
        assert is_private_ip("172.16.0.1")

    def test_link_local(self):
        assert is_private_ip("169.254.169.254")  # AWS metadata

    def test_public_ip(self):
        assert not is_private_ip("8.8.8.8")
        assert not is_private_ip("1.1.1.1")

    def test_invalid_returns_true(self):
        assert is_private_ip("not-an-ip")


class TestMaskUrl:
    def test_long_url_masked(self):
        result = mask_url("https://example.com/very/long/path?token=secret")
        assert "secret" not in result
        assert result.endswith("...")

    def test_short_url(self):
        result = mask_url("https://disk.yandex.ru/d/abc")
        assert "disk.yandex.ru" in result

    def test_invalid_url(self):
        assert mask_url("\x00invalid") == "<invalid-url>" or "invalid" in mask_url("\x00invalid")


class TestValidateExternalUrl:
    def test_non_http_scheme(self):
        assert validate_external_url("file:///etc/passwd") is not None
        assert validate_external_url("gopher://example.com") is not None

    def test_missing_hostname(self):
        assert validate_external_url("http://") is not None

    def test_unresolvable_hostname(self):
        err = validate_external_url("http://nonexistent-host-xyzabc-12345.invalid")
        assert err is not None
