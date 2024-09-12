start-minio:
	minio server ~/minio --console-address :9001
start-server:
	python app/manage.py wait_for_db
	python app/manage.py makemigrations
	python app/manage.py migrate
	python app/manage.py runserver