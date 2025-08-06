from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from fish_app.models import Fish
from shrimp_app.models import Shrimp
from shellfish_app.models import Shellfish
from fish_app.models import StorageCondition as FishStorage
from shrimp_app.models import StorageCondition as ShrimpStorage
from shellfish_app.models import StorageCondition as ShellfishStorage
from fish_app.models import FeatureData as FishData
from shrimp_app.models import FeatureData as ShrimpData
from shellfish_app.models import FeatureData as ShellfishData

def index(request):
    """主页视图"""
    context = {
        'total_species': Fish.objects.count() + Shrimp.objects.count() + Shellfish.objects.count(),
        'fish_count': Fish.objects.count(),
        'shrimp_count': Shrimp.objects.count(),
        'shellfish_count': Shellfish.objects.count(),
    }
    return render(request, 'main_app/index.html', context)

@login_required
def dashboard(request):
    """仪表板视图"""
    # 获取各数据库的统计信息
    context = {
        'total_species': Fish.objects.count() + Shrimp.objects.count() + Shellfish.objects.count(),
        'fish_species_count': Fish.objects.count(),
        'shrimp_species_count': Shrimp.objects.count(),
        'shellfish_species_count': Shellfish.objects.count(),
        
        'total_conditions': (
            FishStorage.objects.count() + 
            ShrimpStorage.objects.count() + 
            ShellfishStorage.objects.count()
        ),
        
        'total_measurements': (
            FishData.objects.count() + 
            ShrimpData.objects.count() + 
            ShellfishData.objects.count()
        ),
        
        'fish_data_points': FishData.objects.count(),
        'shrimp_data_points': ShrimpData.objects.count(),
        'shellfish_data_points': ShellfishData.objects.count(),
        
        # 获取最近更新
        'recent_updates': get_recent_updates()
    }
    
    return render(request, 'main_app/dashboard.html', context)

def get_recent_updates():
    # 这里实现获取最近更新的逻辑
    updates = []
    
    # 获取鱼类最近更新
    fish_updates = Fish.objects.all().order_by('-updated_at')[:5]
    for fish in fish_updates:
        updates.append({
            'database': 'Fish',
            'species': fish.name,
            'time': fish.updated_at,
            'url': f'/fish/{fish.id}/'
        })
    
    # 获取虾类最近更新
    shrimp_updates = Shrimp.objects.all().order_by('-updated_at')[:5]
    for shrimp in shrimp_updates:
        updates.append({
            'database': 'Shrimp',
            'species': shrimp.name,
            'time': shrimp.updated_at,
            'url': f'/shrimp/{shrimp.id}/'
        })
    
    # 获取贝类最近更新
    shellfish_updates = Shellfish.objects.all().order_by('-updated_at')[:5]
    for shellfish in shellfish_updates:
        updates.append({
            'database': 'Shellfish',
            'species': shellfish.name,
            'time': shellfish.updated_at,
            'url': f'/shellfish/{shellfish.id}/'
        })
    
    # 按时间排序并返回最近的10条更新
    updates.sort(key=lambda x: x['time'], reverse=True)
    return updates[:10]

def handler404(request, exception):
    """404错误处理"""
    return render(request, 'main_app/404.html', status=404)

def handler500(request):
    """500错误处理"""
    return render(request, 'main_app/500.html', status=500)
