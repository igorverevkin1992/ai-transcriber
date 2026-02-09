import boto3

from backend.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    BUCKET_NAME,
    REGION,
    logger,
)


def create_s3_client():
    """Создает S3-клиент. Возвращает None если нет ключей."""
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        logger.warning("AWS/S3 ключи не найдены — S3 клиент недоступен.")
        return None
    session = boto3.session.Session()
    return session.client(
        service_name="s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=REGION,
    )


s3 = create_s3_client()


def upload_to_s3(local_path: str, object_name: str) -> str:
    """Загружает файл в S3 и возвращает presigned URL."""
    if not s3:
        raise RuntimeError("S3 клиент не настроен. Проверьте AWS_ACCESS_KEY_ID и AWS_SECRET_ACCESS_KEY.")
    s3.upload_file(local_path, BUCKET_NAME, object_name)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET_NAME, "Key": object_name},
        ExpiresIn=3600,
    )


def delete_from_s3(object_name: str):
    """Удаляет файл из S3."""
    if not s3:
        return
    try:
        s3.delete_object(Bucket=BUCKET_NAME, Key=object_name)
    except Exception as e:
        logger.warning("Не удалось удалить %s из S3: %s", object_name, e)


def check_bucket():
    """Проверяет доступность бакета."""
    if not s3:
        logger.warning("S3 клиент не создан — загрузка файлов в облако невозможна.")
        return
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
        logger.info("S3: бакет '%s' доступен.", BUCKET_NAME)
    except Exception as e:
        logger.error("S3: не удалось получить доступ к бакету '%s': %s", BUCKET_NAME, e)
