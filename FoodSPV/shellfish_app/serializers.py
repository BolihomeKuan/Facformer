from rest_framework import serializers
from .models import ShellfishSpecies, StorageCondition, FeatureType, FeatureData

class ShellfishSpeciesSerializer(serializers.ModelSerializer):
    class Meta:
        model = ShellfishSpecies
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