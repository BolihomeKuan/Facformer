from django.contrib import admin
from .models import Shrimp, StorageCondition, FeatureType, FeatureData

@admin.register(Shrimp)
class ShrimpAdmin(admin.ModelAdmin):
    list_display = ('name', 'scientific_name', 'created_at')
    search_fields = ('name', 'scientific_name')
    list_filter = ('created_at',)

@admin.register(StorageCondition)
class StorageConditionAdmin(admin.ModelAdmin):
    list_display = ('shrimp_species', 'temperature', 'storage_time')
    list_filter = ('shrimp_species', 'temperature')
    search_fields = ('shrimp_species__name',)

@admin.register(FeatureType)
class FeatureTypeAdmin(admin.ModelAdmin):
    list_display = ('name', 'unit')
    search_fields = ('name',)

@admin.register(FeatureData)
class FeatureDataAdmin(admin.ModelAdmin):
    list_display = ('storage_condition', 'feature_type', 'value', 'measurement_day')
    list_filter = ('feature_type', 'storage_condition__shrimp_species')
    search_fields = ('storage_condition__shrimp_species__name', 'feature_type__name')
