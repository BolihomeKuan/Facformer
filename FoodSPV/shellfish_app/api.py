from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import ShellfishSpecies, StorageCondition
from .serializers import (
    ShellfishSpeciesSerializer, 
    StorageConditionSerializer,
    FeatureDataSerializer
)

class ShellfishSpeciesViewSet(viewsets.ModelViewSet):
    queryset = ShellfishSpecies.objects.all()
    serializer_class = ShellfishSpeciesSerializer
    
    @action(detail=True, methods=['get'])
    def storage_conditions(self, request, pk=None):
        shellfish = self.get_object()
        serializer = StorageConditionSerializer(
            shellfish.storage_conditions.all(), 
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