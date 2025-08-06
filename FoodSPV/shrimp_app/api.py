from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import ShrimpSpecies, StorageCondition
from .serializers import (
    ShrimpSpeciesSerializer, 
    StorageConditionSerializer,
    FeatureDataSerializer
)

class ShrimpSpeciesViewSet(viewsets.ModelViewSet):
    queryset = ShrimpSpecies.objects.all()
    serializer_class = ShrimpSpeciesSerializer
    
    @action(detail=True, methods=['get'])
    def storage_conditions(self, request, pk=None):
        shrimp = self.get_object()
        serializer = StorageConditionSerializer(
            shrimp.storage_conditions.all(), 
            many=True
        )
        return Response(serializer.data)

class StorageConditionViewSet(viewsets.ModelViewSet):
    queryset = StorageCondition.objects.all()
    serializer_class = StorageConditionSerializer
    
    @action(detail=True, methods=['get'])
    def feature_data(self, request, pk=None):
        storage = self.get_object()
        serializer = FeatureDataSerializer(
            storage.feature_data.all(), 
            many=True
        )
        return Response(serializer.data) 