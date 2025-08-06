from rest_framework import serializers
from .models import Shrimp, StorageCondition, FeatureType, FeatureData

class ShrimpSerializer(serializers.ModelSerializer):
    class Meta:
        model = Shrimp
        fields = '__all__'

class StorageConditionSerializer(serializers.ModelSerializer):
    class Meta:
        model = StorageCondition
        fields = '__all__'

class FeatureTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = FeatureType
        fields = '__all__'

class FeatureDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = FeatureData
        fields = '__all__' 