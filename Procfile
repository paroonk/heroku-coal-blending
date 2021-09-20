release: python web/manage.py migrate --no-input
web: gunicorn config.wsgi --log-file - --log-level debug
