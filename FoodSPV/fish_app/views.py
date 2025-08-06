from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST, require_http_methods
from django.contrib import messages
from .models import Fish, StorageCondition, FeatureType, FeatureData
from .forms import FishForm, StorageConditionForm, FeatureDataForm, DataImportForm
import numpy as np
from scipy import stats
import pandas as pd
from django import forms
import csv
import json
from django.http import HttpResponse
from datetime import datetime
from .models import Fish, FishSource, StorageCondition, FeatureType, FeatureData
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from .forms import FishForm, BulkFeatureDataForm,StorageConditionForm, FeatureDataForm, DataImportForm, FishSourceForm


@require_http_methods(["GET"])
def index(request):
    """Fish database homepage with both card and table views"""
    fish_species = Fish.objects.all().order_by('name')
    view_type = request.GET.get('view', 'card')  # Default to card view
    
    context = {
        'fish_species': fish_species,
        'view_type': view_type,
        'storage_conditions_count': StorageCondition.objects.count(),
        'feature_data_count': FeatureData.objects.count(),
    }
    return render(request, 'fish_app/index.html', context)

@require_http_methods(["GET"])
@login_required
def dashboard(request):
    """显示仪表板"""
    # 保留原有的统计数据
    total_species = Fish.objects.count()
    total_conditions = StorageCondition.objects.count()
    total_measurements = FeatureData.objects.count()
    feature_types_count = FeatureType.objects.count()

    # 添加鱼类统计数据
    fish_species = Fish.objects.all()
    fish_labels = [fish.name for fish in fish_species]
    fish_measurements = [FeatureData.objects.filter(
        storage_condition__fish_species=fish).count() 
        for fish in fish_species
    ]
    
    # 保留原有的温度和特征类型统计
    temp_data = StorageCondition.objects.values_list('temperature', flat=True)
    temp_labels = sorted(set(temp_data))
    temp_counts = [list(temp_data).count(temp) for temp in temp_labels]

    feature_types = FeatureType.objects.all()
    feature_labels = [ft.name for ft in feature_types]
    feature_data = [FeatureData.objects.filter(feature_type=ft).count() 
                   for ft in feature_types]

    context = {
        'total_species': total_species,
        'total_conditions': total_conditions,
        'total_measurements': total_measurements,
        'feature_types_count': feature_types_count,
        'temp_labels': temp_labels,
        'temp_data': temp_counts,
        'feature_labels': feature_labels,
        'feature_data': feature_data,
        'fish_labels': fish_labels,
        'fish_data': fish_measurements,
    }
    return render(request, 'fish_app/dashboard.html', context)


@login_required
def statistical_analysis(request):
    """统计分析"""
    feature_types = FeatureType.objects.all()
    storage_conditions = StorageCondition.objects.all()
    selected_feature = request.GET.get('feature_type')
    selected_temperature = request.GET.get('temperature')
    
    context = {
        'feature_types': feature_types,
        'storage_conditions': storage_conditions,
        'selected_feature': selected_feature,
        'selected_temperature': selected_temperature
    }

    if selected_feature:
        feature = get_object_or_404(FeatureType, id=selected_feature)
        feature_data_query = FeatureData.objects.filter(feature_type=feature)
        
        if selected_temperature:
            feature_data_query = feature_data_query.filter(
                storage_condition__temperature=selected_temperature
            )
        
        if feature_data_query:
            values = [data.value for data in feature_data_query]
            days = [data.measurement_day for data in feature_data_query]
            
            # 基础统计
            basic_stats = {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            
            # 趋势分析
            slope, intercept, r_value, _, _ = stats.linregress(days, values)
            trend_analysis = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2
            }
            
            context.update({
                'feature': feature,
                'basic_stats': basic_stats,
                'trend_analysis': trend_analysis,
                'days': json.dumps(days),
                'values': json.dumps(values)
            })

    return render(request, 'fish_app/statistical_analysis.html', context)

@login_required
def advanced_search(request):
    fish_list = Fish.objects.all()
    feature_types = FeatureType.objects.all()
    
    context = {
        'fish_list': fish_list,
        'feature_types': feature_types,
    }
    return render(request, 'fish_app/advanced_search.html', context)


def calculate_trend(days, values):
    """计算趋势的更稳健方法"""
    try:
        # 使用 Theil-Sen 估计器，对异常值更稳健
        slope, intercept, _, _ = stats.theilslopes(values, days)
        
        if abs(slope) < 1e-10:  # 接近于零
            return "Stable", slope
        else:
            return "Increasing" if slope > 0 else "Decreasing", slope
    except:
        # 如果数据点太少或有其他问题，使用简单的首尾比较
        if len(values) >= 2:
            trend = "Increasing" if values[-1] > values[0] else "Decreasing"
            slope = (values[-1] - values[0]) / (days[-1] - days[0])
            return trend, slope
        return "Insufficient data", 0

@require_http_methods(["GET", "POST"])
@login_required
def compare_data(request):
    context = {
        'feature_types': FeatureType.objects.all(),
        'storage_conditions': StorageCondition.objects.all()
    }
    
    if request.method == 'POST':
        analysis_type = request.POST.get('analysis_type')
        
        if analysis_type == 'single':
            feature_type_id = request.POST.get('feature_type')
            storage_id = request.POST.get('storage_id')
            
            # 获取单系列数据
            feature_data = FeatureData.objects.filter(
                storage_condition_id=storage_id,
                feature_type_id=feature_type_id
            ).order_by('measurement_day')
            
            if feature_data.exists():
                values = np.array([data.value for data in feature_data])
                days = np.array([data.measurement_day for data in feature_data])
                
                # 确保数据不为空
                if len(values) > 0:
                    # 计算变化率
                    changes = np.diff(values)
                    avg_change_rate = np.mean(changes) if len(changes) > 0 else 0
                    max_change = np.max(np.abs(changes)) if len(changes) > 0 else 0
                    
                    # 计算趋势
                    trend, slope = calculate_trend(days, values)
                    
                    # 准备单系列分析数据
                    single_analysis = {
                        'label': f"{feature_data[0].storage_condition.fish_species.name} ({feature_data[0].storage_condition.temperature}°C)",
                        'days': days.tolist(),
                        'values': values.tolist(),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'median': float(np.median(values)),
                        'q1': float(np.percentile(values, 25)),
                        'q3': float(np.percentile(values, 75)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'trend': trend,
                        'slope': float(slope),
                        'avg_change_rate': float(avg_change_rate),
                        'max_change': float(max_change),
                        'changes': changes.tolist() if len(changes) > 0 else [],
                    }
                    
                    context.update({
                        'single_analysis': single_analysis,
                        'feature_type': FeatureType.objects.get(id=feature_type_id).name
                    })
        
        elif analysis_type == 'compare':
            feature_type_id = request.POST.get('feature_type')
            storage_ids = request.POST.getlist('storage_ids')
            
            # 获取比较数据
            feature_data = FeatureData.objects.filter(
                storage_condition_id__in=storage_ids,
                feature_type_id=feature_type_id
            ).order_by('storage_condition', 'measurement_day')
            
            # 初始化数据结构
            comparison_data = {}
            
            # 处理数据用于图表显示
            for data in feature_data:
                key = f"{data.storage_condition.fish_species.name} ({data.storage_condition.temperature}°C)"
                if key not in comparison_data:
                    comparison_data[key] = {'days': [], 'values': []}
                comparison_data[key]['days'].append(data.measurement_day)
                comparison_data[key]['values'].append(data.value)

            if comparison_data:
                # 准备统计分析结果
                analysis_results = []
                for condition, data in comparison_data.items():
                    values = np.array(data['values'])
                    mean = np.mean(values)
                    cv = (np.std(values) / mean * 100) if mean != 0 else 0
                    
                    analysis_results.append({
                        'condition': condition,
                        'mean': mean,
                        'cv': cv,
                        'values': values
                    })

                # 找出最佳条件
                best_condition = max(analysis_results, key=lambda x: x['mean'])
                most_stable = min(analysis_results, key=lambda x: x['cv'])

                # ANOVA 分析
                anova_result = None
                if len(comparison_data) >= 2:
                    try:
                        groups = [result['values'] for result in analysis_results]
                        f_stat, p_value = stats.f_oneway(*groups)
                        anova_result = {
                            'f_stat': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except:
                        anova_result = {'error': 'Could not perform ANOVA analysis'}

                # 准备图表数据
                chart_datasets = []
                colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
                for i, (condition, data) in enumerate(comparison_data.items()):
                    chart_datasets.append({
                        'label': condition,
                        'data': data['values'],
                        'borderColor': colors[i % len(colors)],
                        'fill': False
                    })

                # 准备柱状图数据
                bar_labels = []
                bar_values = []
                for result in analysis_results:
                    bar_labels.append(result['condition'])
                    bar_values.append(float(result['mean']))  # 确保数值可以被JSON序列化

                context.update({
                    'comparison_data': comparison_data,
                    'analysis_results': analysis_results,
                    'best_condition': best_condition['condition'],
                    'most_stable_condition': most_stable['condition'],
                    'anova_result': anova_result,
                    'chart_datasets': json.dumps(chart_datasets),
                    'days': json.dumps(list(range(1, max(max(d['days']) for d in comparison_data.values()) + 1))),
                    'feature_type': FeatureType.objects.get(id=feature_type_id).name,
                    'bar_labels': json.dumps(bar_labels),
                    'bar_values': json.dumps(bar_values)
                })
    
    return render(request, 'fish_app/compare_data.html', context)


@require_http_methods(["POST"])
@login_required
def compare_analysis(request):
    """对比分析"""
    feature_type_id = request.POST.get('feature_type')
    condition_ids = request.POST.getlist('conditions[]')
    
    if not feature_type_id or len(condition_ids) < 2:
        messages.error(request, 'Please select a feature type and at least two storage conditions for comparison')
        return redirect('fish_app:statistical_analysis')
        
    feature_type = get_object_or_404(FeatureType, id=feature_type_id)
    comparison_data = []
    comparison_datasets = []
    comparison_stats_datasets = []
    colors = [
        'rgba(75, 192, 192, 0.6)',
        'rgba(255, 99, 132, 0.6)',
        'rgba(255, 205, 86, 0.6)',
        'rgba(54, 162, 235, 0.6)',
        'rgba(153, 102, 255, 0.6)'
    ]
    
    all_days = set()
    for condition_id in condition_ids:
        storage = get_object_or_404(StorageCondition, id=condition_id)
        data = FeatureData.objects.filter(
            storage_condition=storage,
            feature_type=feature_type
        ).order_by('measurement_day')
        
        if data:
            days = [d.measurement_day for d in data]
            values = [d.value for d in data]
            all_days.update(days)
            
            comparison_datasets.append({
                'label': f'{storage.fish_species.name} ({storage.temperature}°C)',
                'data': values,
                'borderColor': colors[len(comparison_datasets)],
                'backgroundColor': colors[len(comparison_datasets)],
                'fill': False
            })
            
            comparison_stats_datasets.append({
                'label': f'{storage.fish_species.name} ({storage.temperature}°C)',
                'data': [
                    float(np.mean(values)),
                    float(np.median(values)),
                    float(np.std(values))
                ],
                'backgroundColor': colors[len(comparison_stats_datasets)]
            })
            
            comparison_data.append({
                'storage': storage,
                'data': data
            })
    
    if not comparison_data:
        messages.error(request, 'No available data for the selected conditions')
        return redirect('fish_app:statistical_analysis')
    
    context = {
        'feature_type': feature_type,
        'comparison_data': comparison_data,
        'days': json.dumps(sorted(list(all_days))),
        'comparison_datasets': json.dumps(comparison_datasets),
        'comparison_stats_datasets': json.dumps(comparison_stats_datasets)
    }
    
    return render(request, 'fish_app/statistical_analysis.html', context)










    
@require_http_methods(["GET"])
def fish_detail(request, fish_id):
    """Display detailed information for a specific fish species"""
    fish = get_object_or_404(Fish, id=fish_id)
    sources = fish.sources.all().prefetch_related('storage_conditions')
    
    context = {
        'fish': fish,
        'sources': sources
    }
    return render(request, 'fish_app/fish_detail.html', context)

@require_http_methods(["GET"])
def fish_list(request):
    """Display fish species in table format"""
    fish_species = Fish.objects.all().order_by('name')
    context = {
        'fish_species': fish_species,
        'storage_conditions_count': StorageCondition.objects.count(),
        'feature_data_count': FeatureData.objects.count(),
    }
    return render(request, 'fish_app/fish_list.html', context)


@require_http_methods(["GET", "POST"])
@login_required
def add_fish(request):
    """添加新的鱼类"""
    if request.method == 'POST':
        form = FishForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully added new fish species.')
            return redirect('fish_app:index')
    else:
        form = FishForm()
    context = {'form': form}
    return render(request, 'fish_app/add_fish.html', context)

@require_http_methods(["GET", "POST"])
@login_required
def edit_fish(request, fish_id):
    """编辑鱼类信息"""
    fish = get_object_or_404(Fish, id=fish_id)
    if request.method == 'POST':
        form = FishForm(request.POST, request.FILES, instance=fish)
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully updated fish species.')
            return redirect('fish_app:fish_detail', fish_id=fish.id)
    else:
        form = FishForm(instance=fish)
    context = {'form': form, 'fish': fish}
    return render(request, 'fish_app/edit_fish.html', context)

@login_required
def delete_confirm(request, fish_id):
    fish = get_object_or_404(Fish, pk=fish_id)
    return render(request, 'fish_app/delete_confirm.html', {'fish': fish})

@login_required
@require_POST
def delete_fish(request, fish_id):
    fish = get_object_or_404(Fish, pk=fish_id)
    fish.delete()
    return redirect('fish_app:fish_list')





@require_http_methods(["GET", "POST"])
@login_required
def add_storage(request, fish_id, source_id):
    """添加储存条件"""
    fish = get_object_or_404(Fish, id=fish_id)
    source = get_object_or_404(FishSource, id=source_id)
    
    if request.method == 'POST':
        form = StorageConditionForm(request.POST)
        if form.is_valid():
            storage = form.save(commit=False)
            storage.fish_species = fish
            storage.fish_source = source
            storage.save()
            messages.success(request, 'Successfully added storage condition.')
            return redirect('fish_app:fish_source_detail', fish_id=fish_id, source_id=source_id)
    else:
        form = StorageConditionForm()
    
    context = {
        'form': form, 
        'fish': fish,
        'source': source
    }
    return render(request, 'fish_app/add_storage.html', context)

@require_http_methods(["GET", "POST"])
@login_required
def edit_storage(request, storage_id):
    """编辑储存条件"""
    storage = get_object_or_404(StorageCondition, id=storage_id)
    if request.method == 'POST':
        form = StorageConditionForm(request.POST, instance=storage)
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully updated storage condition.')
            return redirect('fish_app:storage_detail', storage_id=storage.id)
    else:
        form = StorageConditionForm(instance=storage)
    context = {'form': form, 'storage': storage}
    return render(request, 'fish_app/edit_storage.html', context)

@require_http_methods(["GET"])
def storage_detail(request, storage_id):
    """显示储存条件详情"""
    storage = get_object_or_404(StorageCondition, id=storage_id)
    
    # 获取所有特征类型
    feature_types = FeatureType.objects.filter(
        featuredata__storage_condition=storage
    ).distinct()
    
    # 获取选中的特征类型
    selected_feature = request.GET.get('feature_type')
    
    # 基础查询
    feature_data_query = FeatureData.objects.filter(storage_condition=storage)
    
    # 如果选择了特征类型，进行筛选
    if selected_feature:
        feature_data_query = feature_data_query.filter(feature_type_id=selected_feature)
    
    context = {
        'storage': storage,
        'feature_data': feature_data_query,
        'feature_types': feature_types,
        'selected_feature': selected_feature
    }
    return render(request, 'fish_app/storage_detail.html', context)

@require_http_methods(["POST"])
@login_required
def delete_storage(request, storage_id):
    """删除储存条件"""
    storage = get_object_or_404(StorageCondition, id=storage_id)
    fish_id = storage.fish_species.id
    storage.delete()
    messages.success(request, 'Successfully deleted storage condition.')
    return redirect('fish_app:fish_detail', fish_id=fish_id)







@require_http_methods(["GET", "POST"])
@login_required
def add_feature_data(request, storage_id):
    """添加特征数据"""
    storage = get_object_or_404(StorageCondition, id=storage_id)
    if request.method == 'POST':
        form = FeatureDataForm(request.POST)
        if form.is_valid():
            feature_data = form.save(commit=False)
            feature_type_name = form.cleaned_data['feature_type']
            
            # 获取或创建特征类型
            feature_type, created = FeatureType.objects.get_or_create(
                name=feature_type_name
            )
            
            feature_data.feature_type = feature_type
            feature_data.storage_condition = storage
            feature_data.save()
            
            messages.success(request, 'Successfully added feature data.')
            return redirect('fish_app:storage_detail', storage_id=storage.id)
    else:
        form = FeatureDataForm()
    
    context = {'form': form, 'storage': storage}
    return render(request, 'fish_app/add_feature_data.html', context)
@require_http_methods(["POST"])
@login_required
def delete_feature_data(request, feature_id):
    feature = get_object_or_404(FeatureData, pk=feature_id)
    storage_id = feature.storage_condition.id
    feature.delete()
    messages.success(request, 'Feature data deleted successfully.')
    return redirect('fish_app:storage_detail', storage_id=storage_id)

@require_http_methods(["GET"])
def feature_detail(request, storage_id, feature_id):
    """显示特征数详情"""
    storage = get_object_or_404(StorageCondition, id=storage_id)
    feature_type = get_object_or_404(FeatureType, id=feature_id)
    feature_data = FeatureData.objects.filter(
        storage_condition=storage,
        feature_type=feature_type
    ).order_by('measurement_day')
    
    context = {
        'storage': storage,
        'feature_type': feature_type,
        'feature_data': feature_data
    }
    return render(request, 'fish_app/feature_detail.html', context)


@login_required
def edit_feature_data(request, feature_id):
    feature_data = get_object_or_404(FeatureData, pk=feature_id)
    
    if request.method == 'POST':
        form = FeatureDataForm(request.POST, instance=feature_data)
        if form.is_valid():
            form.save()
            return redirect('fish_app:storage_detail', storage_id=feature_data.storage_condition.id)
    else:
        form = FeatureDataForm(instance=feature_data)
    
    context = {
        'form': form,
        'feature_data': feature_data,
        'storage': feature_data.storage_condition
    }
    return render(request, 'fish_app/edit_feature_data.html', context)
@require_http_methods(["GET", "POST"])
@login_required
def bulk_add_feature_data(request, storage_id):
    """Bulk add feature data"""
    storage = get_object_or_404(StorageCondition, id=storage_id)
    
    if request.method == 'POST':
        form = BulkFeatureDataForm(request.POST)
        if form.is_valid():
            bulk_data = form.cleaned_data['bulk_data']
            lines = bulk_data.strip().split('\n')
            success_count = 0
            error_messages = []
            
            for line in lines:
                parts = line.strip().split(None, 3)  # Split into max 4 parts
                if len(parts) < 3:
                    error_messages.append(f'Invalid line format: {line}')
                    continue
                    
                feature_type_name = parts[0]
                try:
                    measurement_day = int(parts[1])
                    value = float(parts[2])
                    notes = parts[3] if len(parts) > 3 else ''
                    
                    # Get or create feature type
                    feature_type, _ = FeatureType.objects.get_or_create(
                        name=feature_type_name
                    )
                    
                    # Create feature data
                    FeatureData.objects.create(
                        storage_condition=storage,
                        feature_type=feature_type,
                        measurement_day=measurement_day,
                        value=value,
                        notes=notes
                    )
                    success_count += 1
                except ValueError as e:
                    error_messages.append(f'Error in line: {line} - {str(e)}')
            
            if success_count > 0:
                messages.success(request, f'Successfully added {success_count} measurements.')
            if error_messages:
                messages.warning(request, 'Some entries had errors: ' + '; '.join(error_messages))
            return redirect('fish_app:storage_detail', storage_id=storage.id)
    else:
        form = BulkFeatureDataForm()
    
    context = {
        'form': form,
        'storage': storage
    }
    return render(request, 'fish_app/bulk_add_feature_data.html', context)


@require_http_methods(["GET"])
def export_feature_data(request, storage_id):
    """Export selected feature data to CSV"""
    storage = get_object_or_404(StorageCondition, id=storage_id)
    selected_feature = request.GET.get('feature_type')
    
    # Create the HttpResponse object with CSV header
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="feature_data_{storage.id}.csv"'
    
    writer = csv.writer(response)
    writer.writerow([
        'Fish Name',
        'Scientific Name',
        'Feature Type',
        'Day',
        'Value',
        'Unit',
        'Temperature',
        'Notes'
    ])
    
    # Base query
    feature_data_query = FeatureData.objects.filter(storage_condition=storage)
    
    # Apply feature type filter if selected
    if selected_feature:
        feature_data_query = feature_data_query.filter(feature_type_id=selected_feature)
    
    # Write data rows
    for data in feature_data_query:
        writer.writerow([
            storage.fish_species.name,
            storage.fish_species.scientific_name,
            data.feature_type.name,
            data.measurement_day,
            data.value,
            data.feature_type.unit,
            storage.temperature,  # 移除℃符号
            data.notes
        ])
    
    return response





@require_http_methods(["GET", "POST"])
@login_required
def import_data(request):
    """导入数据"""
    if request.method == 'POST':
        form = DataImportForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                file = request.FILES['file']
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file, sheet_name=form.cleaned_data['sheet_name'])
                
                for _, row in df.iterrows():
                    fish, _ = Fish.objects.get_or_create(
                        name=row['Fish Species'],
                        defaults={
                            'scientific_name': row.get('Scientific Name', ''),
                            'description': row.get('Description', '')
                        }
                    )
                    
                    storage, _ = StorageCondition.objects.get_or_create(
                        fish_species=fish,
                        temperature=row['Temperature'],
                        defaults={
                            'storage_time': row['Storage Time'],
                            'notes': row.get('Notes', '')
                        }
                    )
                    
                    feature_type, _ = FeatureType.objects.get_or_create(
                        name=row['Feature'],
                        defaults={
                            'unit': row.get('Unit', ''),
                            'description': row.get('Feature Description', '')
                        }
                    )
                    
                    FeatureData.objects.create(
                        storage_condition=storage,
                        feature_type=feature_type,
                        value=row['Value'],
                        measurement_day=row['Day']
                    )
                
                messages.success(request, 'Data imported successfully!')
                return redirect('fish_app:dashboard')
            except Exception as e:
                messages.error(request, f'Error importing data: {str(e)}')
    else:
        form = DataImportForm()
    
    return render(request, 'fish_app/import_data.html', {'form': form})






@require_http_methods(["GET", "POST"])
@login_required
def bulk_edit(request):
    """批量编辑数据"""
    if request.method == 'POST':
        action = request.POST.get('action')
        selected_items = request.POST.getlist('selected_items')
        
        if not selected_items:
            messages.error(request, 'No items selected.')
            return redirect('fish_app:bulk_edit')
            
        if action == 'delete':
            FeatureData.objects.filter(id__in=selected_items).delete()
            messages.success(request, f'Successfully deleted {len(selected_items)} items.')
            
        elif action == 'update':
            try:
                adjustment = float(request.POST.get('value_adjustment', 0))
                items = FeatureData.objects.filter(id__in=selected_items)
                for item in items:
                    item.value += adjustment
                    item.save()
                messages.success(request, f'Successfully updated {len(selected_items)} items.')
            except ValueError:
                messages.error(request, 'Invalid adjustment value.')
                
        return redirect('fish_app:bulk_edit')
        
    feature_data = FeatureData.objects.select_related(
        'storage_condition__fish_species',
        'feature_type'
    ).all()
    
    context = {
        'feature_data': feature_data
    }
    return render(request, 'fish_app/bulk_edit.html', context)





@login_required
def export_data(request, fish_id):
    """导出鱼类数据"""
    fish = get_object_or_404(Fish, id=fish_id)
    
    # 创建 HTTP 响应对象，设置文件类型和文件名
    response = HttpResponse(content_type='text/csv')
    filename = f"{fish.name}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    # 创建 CSV 写入器
    writer = csv.writer(response)
    
    # 写入表头
    writer.writerow([
        'Fish Species',
        'Scientific Name',
        'Temperature (°C)',
        'Storage Time (days)',
        'Feature',
        'Value',
        'Unit',
        'Measurement Day',
        'Notes'
    ])
    
    # 获取所有相关数据
    storage_conditions = StorageCondition.objects.filter(fish_species=fish)
    for storage in storage_conditions:
        feature_data = FeatureData.objects.filter(storage_condition=storage)
        for data in feature_data:
            writer.writerow([
                fish.name,
                fish.scientific_name,
                storage.temperature,
                storage.storage_time,
                data.feature_type.name,
                data.value,
                data.feature_type.unit,
                data.measurement_day,
                data.notes
            ])
    
    return response








@require_http_methods(["GET"])
def fish_source_detail(request, fish_id, source_id):
    """View for fish source details"""
    fish = get_object_or_404(Fish, id=fish_id)
    source = get_object_or_404(FishSource, id=source_id)
    storage_conditions = StorageCondition.objects.filter(
        fish_species=fish,
        fish_source=source
    ).select_related('fish_species', 'fish_source')
    
    storage_with_features = []
    for storage in storage_conditions:
        feature_types = FeatureType.objects.filter(
            featuredata__storage_condition=storage
        ).distinct()
        
        # 获取该储存条件下的所有数据点数量
        data_points_count = FeatureData.objects.filter(
            storage_condition=storage
        ).count()
        
        storage_with_features.append({
            'storage': storage,
            'feature_types': feature_types,
            'data_points_count': data_points_count  # 添加数据点数量
        })
    
    context = {
        'fish': fish,
        'source': source,
        'storage_with_features': storage_with_features
    }
    return render(request, 'fish_app/fish_source_detail.html', context)




@require_http_methods(["GET", "POST"])
@login_required
def add_source(request, fish_id):
    """添加鱼类来源"""
    fish = get_object_or_404(Fish, id=fish_id)
    
    if request.method == 'POST':
        form = FishSourceForm(request.POST)
        if form.is_valid():
            source = form.save(commit=False)
            source.fish_species = fish
            source.save()
            messages.success(request, 'Successfully added fish source.')
            return redirect('fish_app:fish_detail', fish_id=fish.id)
    else:
        form = FishSourceForm()
        # 移除这行代码，因为fish_species字段应该在表单的Meta类中被排除
        # form.fields['fish_species'].widget = forms.HiddenInput()
    
    context = {'form': form, 'fish': fish}
    return render(request, 'fish_app/add_source.html', context)

@require_http_methods(["GET", "POST"])
@login_required
def edit_source(request, source_id):
    """Edit fish source"""
    source = get_object_or_404(FishSource, id=source_id)
    fish_id = source.fish_species.id
    if request.method == 'POST':
        form = FishSourceForm(request.POST, instance=source)
        if form.is_valid():
            form.save()
            messages.success(request, 'Successfully updated source.')
            return redirect('fish_app:fish_source_detail', fish_id=fish_id, source_id=source.id)
    else:
        form = FishSourceForm(instance=source)
    context = {'form': form, 'source': source, 'fish': source.fish_species}  # 修改这里
    return render(request, 'fish_app/edit_source.html', context)


@require_http_methods(["POST"])
@login_required
def delete_source(request, source_id):
    """Delete fish source"""
    source = get_object_or_404(FishSource, id=source_id)
    fish_id = source.fish_species.id
    source.delete()
    messages.success(request, 'Successfully deleted source.')
    return redirect('fish_app:fish_detail', fish_id=fish_id)





