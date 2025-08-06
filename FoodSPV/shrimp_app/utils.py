from django.core.mail import send_mail
from django.conf import settings
from django.contrib.auth.models import User
from .models import (
    FeatureData, 
    StorageCondition, 
    Shrimp, 
    FeatureType
)
import numpy as np

def validate_data_quality(value, feature_type, storage_condition):
    """数据质量控制"""
    # 获取该特征类型的历史数据
    historical_data = FeatureData.objects.filter(
        feature_type=feature_type,
        storage_condition__temperature=storage_condition.temperature
    ).values_list('value', flat=True)
    
    if historical_data:
        mean = np.mean(historical_data)
        std = np.std(historical_data)
        
        # 检查是否在3个标准差范围内
        if abs(value - mean) > 3 * std:
            return False, f"Warning: Value {value} is outside normal range (mean: {mean:.2f}, std: {std:.2f})"
    
    return True, "Data quality check passed"

def send_notification(subject, message, recipient_list):
    """发送邮件通知"""
    send_mail(
        subject=subject,
        message=message,
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=recipient_list,
        fail_silently=False,
    )

def notify_data_change(instance, created, **kwargs):
    """数据变更通知"""
    if created:
        subject = f'New {instance.__class__.__name__} Added'
        action = 'added'
    else:
        subject = f'{instance.__class__.__name__} Updated'
        action = 'updated'
    
    message = f"""
    A {instance.__class__.__name__} has been {action}:
    
    Details:
    {str(instance)}
    
    Please review the changes if necessary.
    """
    
    # 获取需要通知的用户邮箱
    recipients = User.objects.filter(
        userpreference__email_notifications=True
    ).values_list('email', flat=True)
    
    send_notification(subject, message, recipients)

def backup_database(request):
    """备份数据库"""
    from django.core import serializers
    import json
    from datetime import datetime
    import os
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = os.path.join(settings.BASE_DIR, 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    
    backup_data = {
        'shrimp_species': serializers.serialize('json', Shrimp.objects.all()),
        'storage_conditions': serializers.serialize('json', StorageCondition.objects.all()),
        'feature_types': serializers.serialize('json', FeatureType.objects.all()),
        'feature_data': serializers.serialize('json', FeatureData.objects.all()),
    }
    
    filename = os.path.join(backup_dir, f'backup_{timestamp}.json')
    with open(filename, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    return filename

def restore_database(backup_file):
    """从备份恢复数据库"""
    import json
    from django.core.serializers import deserialize
    
    with open(backup_file, 'r') as f:
        backup_data = json.load(f)
    
    # 清除现有数据
    FeatureData.objects.all().delete()
    StorageCondition.objects.all().delete()
    Shrimp.objects.all().delete()
    FeatureType.objects.all().delete()
    
    # 恢复数据
    for model_name, data in backup_data.items():
        for obj in deserialize('json', data):
            obj.save() 