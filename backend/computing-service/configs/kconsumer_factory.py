import json

from aiokafka import AIOKafkaConsumer


class KConsumerFactory:
    DEFAULT_BOOTSTRAP_SERVERS = 'localhost:9094'
    DEFAULT_GROUP_ID = 'fastapi-group'

    @staticmethod
    def create_consumer(topic: str,
                        bootstrap_servers: str = DEFAULT_BOOTSTRAP_SERVERS,
                        group_id: str = DEFAULT_GROUP_ID) -> AIOKafkaConsumer:
        return AIOKafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
