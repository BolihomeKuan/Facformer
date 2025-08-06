from django.urls import path
from . import views

app_name = 'fish_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('compare-data/', views.compare_data, name='compare_data'),
    path('comparison-result/', views.compare_analysis, name='comparison_result'),
    path('advanced-search/', views.advanced_search, name='advanced_search'),
    path('statistical-analysis/', views.statistical_analysis, name='statistical_analysis'),
    
    # Fish Management
    path('add/', views.add_fish, name='add_fish'),
    path('<int:fish_id>/edit/', views.edit_fish, name='edit_fish'),
    path('<int:fish_id>/delete/confirm/', views.delete_confirm, name='delete_confirm'),
    path('<int:fish_id>/delete/', views.delete_fish, name='delete_fish'),
    path('<int:fish_id>/', views.fish_detail, name='fish_detail'),
    path('list/', views.fish_list, name='fish_list'),
    
    # Source Management 
    path('<int:fish_id>/source/<int:source_id>/', views.fish_source_detail, name='fish_source_detail'),
    path('<int:fish_id>/add-source/', views.add_source, name='add_source'),
    path('source/<int:source_id>/edit/', views.edit_source, name='edit_source'),
    path('source/<int:source_id>/delete/', views.delete_source, name='delete_source'),

    # Storage Management
    path('<int:fish_id>/source/<int:source_id>/add-storage/', views.add_storage, name='add_storage'),
    path('storage/<int:storage_id>/', views.storage_detail, name='storage_detail'),
    path('storage/<int:storage_id>/edit/', views.edit_storage, name='edit_storage'),
    path('storage/<int:storage_id>/delete/', views.delete_storage, name='delete_storage'),
    
    # Feature Data Management
    path('feature/<int:feature_id>/edit/', views.edit_feature_data, name='edit_feature_data'),
    path('feature/<int:feature_id>/delete/', views.delete_feature_data, name='delete_feature_data'),
    path('storage/<int:storage_id>/feature/<int:feature_id>/delete/', views.delete_feature_data, name='delete_feature'),
    path('storage/<int:storage_id>/add-feature/', views.add_feature_data, name='add_feature_data'),
    path('storage/<int:storage_id>/bulk-add-feature/', views.bulk_add_feature_data, name='bulk_add_feature_data'),
    
    # Data Import and Bulk Edit
    path('import/', views.import_data, name='import_data'),
    path('bulk-edit/', views.bulk_edit, name='bulk_edit'),
    path('storage/<int:storage_id>/export/', views.export_feature_data, name='export_feature_data'),
]
