from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import FishSpecies, StorageCondition
from .serializers import (
    FishSpeciesSerializer, 
    StorageConditionSerializer,
    FeatureDataSerializer
)

class FishSpeciesViewSet(viewsets.ModelViewSet):
    queryset = FishSpecies.objects.all()
    serializer_class = FishSpeciesSerializer
    
    @action(detail=True, methods=['get'])
    def storage_conditions(self, request, pk=None):
        fish = self.get_object()
        serializer = StorageConditionSerializer(
            fish.storage_conditions.all(), 
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