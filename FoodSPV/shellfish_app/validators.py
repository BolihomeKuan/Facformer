from django.core.exceptions import ValidationError
import numpy as np
import pandas as pd
from shellfish_app.models import FeatureData

def validate_temperature(value):
    """验证温度值是否在合理范围内"""
    if value < -40 or value > 40:
        raise ValidationError('Temperature must be between -40°C and 40°C')

def validate_measurement_value(value, feature_type, storage_condition):
    """验证测量值是否在合理范围内"""
    # 获取历史数据
    historical_data = FeatureData.objects.filter(
        feature_type=feature_type,
        storage_condition__temperature=storage_condition.temperature
    ).values_list('value', flat=True)
    
    if historical_data:
        mean = np.mean(historical_data)
        std = np.std(historical_data)
        if abs(value - mean) > 3 * std:
            raise ValidationError(
                f'Value {value} is outside normal range (mean: {mean:.2f}, std: {std:.2f})'
            )

def clean_data(df):
    """清理导入的数据"""
    # 移除重复行
    df = df.drop_duplicates()
    
    # 填充缺失值
    df['Notes'] = df['Notes'].fillna('')
    df['Description'] = df['Description'].fillna('')
    
    # 数值列转换
    df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
    
    # 移除无效数据
    df = df.dropna(subset=['Temperature', 'Value', 'Day'])
    
    return df 

def validate_import_data(df):
    """验证导入的数据"""
    errors = []
    
    # 检查必需列
    required_columns = ['Shellfish Species', 'Temperature', 'Storage Time', 
                       'Feature', 'Value', 'Day']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # 验证数值范围
    if 'Temperature' in df.columns:
        invalid_temps = df[
            (df['Temperature'] < -40) | (df['Temperature'] > 40)
        ]['Temperature'].tolist()
        if invalid_temps:
            errors.append(f"Invalid temperatures found: {invalid_temps}")
    
    if 'Day' in df.columns:
        invalid_days = df[df['Day'] < 0]['Day'].tolist()
        if invalid_days:
            errors.append(f"Invalid days found: {invalid_days}")
    
    # 检查数据一致性
    if 'Shellfish Species' in df.columns and 'Temperature' in df.columns:
        duplicates = df.groupby(['Shellfish Species', 'Temperature', 'Day', 'Feature'])\
            .size().reset_index(name='count')
        duplicates = duplicates[duplicates['count'] > 1]
        if not duplicates.empty:
            errors.append("Duplicate measurements found for same conditions")
    
    return errors 