from django import template
import json

register = template.Library()

@register.filter(name='format_float')
def format_float(value, decimals=2):
    """格式化浮点数"""
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return value

@register.filter(name='format_percentage')
def format_percentage(value, decimals=1):
    """将小数转换为百分比"""
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return ''

@register.filter(name='format_temperature')
def format_temperature(value):
    """格式化温度显示"""
    try:
        return f"{float(value):.1f}°C"
    except (ValueError, TypeError):
        return value

@register.filter(name='to_json')
def to_json(value):
    """将Python对象转换为JSON字符串"""
    return json.dumps(value)

@register.filter(name='get_dict_value')
def get_dict_value(dictionary, key):
    """从字典中获取值"""
    return dictionary.get(key) 