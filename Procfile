release: python web/manage.py migrate --no-input
web: gunicorn web.config.wsgi --log-file - --log-level debug
