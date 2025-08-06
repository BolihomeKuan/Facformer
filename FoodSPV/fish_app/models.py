from django.db import models
from django.utils import timezone

class Fish(models.Model):
    name = models.CharField(max_length=100)
    scientific_name = models.CharField(max_length=100, blank=True)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='fish_images/', blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
 
    class Meta:
        verbose_name_plural = "Fish Species"

    def __str__(self):
        return self.name

class FishSource(models.Model):
    SOURCE_TYPE_CHOICES = [
        ('EXPERIMENT', 'Experimental Test'),
        ('LITERATURE', 'Literature Reference'),
    ]
    
    fish_species = models.ForeignKey(Fish, on_delete=models.CASCADE, related_name='sources')
    source_code = models.CharField(max_length=50, verbose_name='Source Code')
    source_type = models.CharField(
        max_length=20,
        choices=SOURCE_TYPE_CHOICES,
        verbose_name='Source Type'
    )
    source_name = models.CharField(max_length=200, verbose_name='Source Name')
    source_date = models.DateField(verbose_name='Data Collection Date')
    description = models.TextField(blank=True, verbose_name='Description')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Fish Source'
        verbose_name_plural = 'Fish Sources'
        ordering = ['-source_date']

    def __str__(self):
        return f"{self.fish_species.name} - {self.source_name}"

class StorageCondition(models.Model):
    fish_species = models.ForeignKey(Fish, related_name='storage_conditions', on_delete=models.CASCADE)
    fish_source = models.ForeignKey(
        FishSource, 
        related_name='storage_conditions', 
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    temperature = models.FloatField()
    storage_time = models.IntegerField()  # Storage time in days
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['temperature']
        verbose_name = 'Storage Condition'
        verbose_name_plural = 'Storage Conditions'

    def __str__(self):
        source_name = self.fish_source.source_name if self.fish_source else 'Unknown'
        return f"{self.fish_species.name} ({source_name}) at {self.temperature}°C"

class FeatureType(models.Model):
    name = models.CharField(max_length=100)
    unit = models.CharField(max_length=50)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.unit})"

class FeatureData(models.Model):
    storage_condition = models.ForeignKey(StorageCondition, related_name='feature_data', on_delete=models.CASCADE)
    feature_type = models.ForeignKey(FeatureType, on_delete=models.CASCADE)
    value = models.FloatField()
    measurement_day = models.IntegerField()
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['measurement_day']

    def __str__(self):
        return f"{self.feature_type.name}: {self.value} {self.feature_type.unit}"