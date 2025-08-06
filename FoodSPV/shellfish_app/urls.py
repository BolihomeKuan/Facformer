from django.urls import path
from . import views

app_name = 'shellfish_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('compare-data/', views.compare_data, name='compare_data'),
    path('comparison-result/', views.comparison_result, name='comparison_result'),
    path('advanced-search/', views.advanced_search, name='advanced_search'),
    path('statistical-analysis/', views.statistical_analysis, name='statistical_analysis'),
    
    # 贝类管理
    path('add/', views.add_shellfish, name='add_shellfish'),
    path('<int:shellfish_id>/edit/', views.edit_shellfish, name='edit_shellfish'),
    path('<int:shellfish_id>/delete/confirm/', views.delete_confirm, name='delete_confirm'),
    path('<int:shellfish_id>/delete/', views.delete_shellfish, name='delete_shellfish'),
    path('<int:shellfish_id>/', views.shellfish_detail, name='shellfish_detail'),
    path('list/', views.shellfish_list, name='shellfish_list'),
    
    # 来源管理
    path('<int:shellfish_id>/source/<int:source_id>/', views.shellfish_source_detail, name='shellfish_source_detail'),
    path('<int:shellfish_id>/add-source/', views.add_source, name='add_source'),
    path('source/<int:source_id>/edit/', views.edit_source, name='edit_source'),
    path('source/<int:source_id>/delete/', views.delete_source, name='delete_source'),

    # 存储条件管理
    path('<int:shellfish_id>/source/<int:source_id>/add-storage/', views.add_storage, name='add_storage'),
    path('storage/<int:storage_id>/', views.storage_detail, name='storage_detail'),
    path('storage/<int:storage_id>/edit/', views.edit_storage, name='edit_storage'),
    path('storage/<int:storage_id>/delete/', views.delete_storage, name='delete_storage'),
    
    # 特征数据管理
    path('storage/<int:storage_id>/feature/<int:data_id>/edit/', views.edit_feature_data, name='edit_feature_data'),
    path('storage/<int:storage_id>/feature/<int:feature_id>/delete/', views.delete_feature_data, name='delete_feature_data'),
    path('storage/<int:storage_id>/add-feature/', views.add_feature_data, name='add_feature_data'),
    path('storage/<int:storage_id>/bulk-add-feature/', views.bulk_add_feature_data, name='bulk_add_feature_data'),
    
    # 数据导入和批量编辑
    path('import/', views.import_data, name='import_data'),
    path('bulk-edit/', views.bulk_edit, name='bulk_edit'),
    path('storage/<int:storage_id>/export/', views.export_feature_data, name='export_feature_data'), 
]
