rabbitmq-server &

pid=$!

sleep 10

echo setting permissions
rabbitmqctl add_user celery celery
rabbitmqctl add_vhost celery_host
rabbitmqctl set_user_tags celery airflow
rabbitmqctl set_permissions -p celery_host celery ".*" ".*" ".*"
echo "Done!"

wait $pid