import time
import logging
from django.db import connection

logger = logging.getLogger(__name__)

class PerformanceMonitorMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start_time = time.time()
        
        # 记录数据库查询数量
        initial_queries = len(connection.queries)
        
        response = self.get_response(request)
        
        # 计算执行时间和查询数量
        execution_time = time.time() - start_time
        queries_count = len(connection.queries) - initial_queries
        
        # 记录性能数据
        logger.info(
            f'Path: {request.path} | '
            f'Time: {execution_time:.2f}s | '
            f'Queries: {queries_count}'
        )
        
        return response 