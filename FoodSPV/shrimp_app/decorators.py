from functools import wraps
from .models import UserActivityLog

def log_activity(action):
    def decorator(view_func):
        @wraps(view_func)
        def _wrapped_view(request, *args, **kwargs):
            response = view_func(request, *args, **kwargs)
            if request.user.is_authenticated:
                UserActivityLog.objects.create(
                    user=request.user,
                    action=action,
                    target=request.path,
                    details=str(request.POST or request.GET),
                    ip_address=request.META.get('REMOTE_ADDR')
                )
            return response
        return _wrapped_view
    return decorator 