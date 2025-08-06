from django.db import models
from django.utils import timezone

class Shellfish(models.Model):
    name = models.CharField(max_length=100)
    scientific_name = models.CharField(max_length=100, blank=True)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='shellfish_images/', blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
 
    class Meta:
        verbose_name_plural = "Shellfish Species"

    def __str__(self):
        return self.name

class ShellfishSource(models.Model):
    SOURCE_TYPE_CHOICES = [
        ('ENTERPRISE', 'Enterprise Provided'),
        ('LITERATURE', 'Literature Reference'),
    ]
    
    shellfish_species = models.ForeignKey(Shellfish, on_delete=models.CASCADE, related_name='sources')
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
        verbose_name = 'Shellfish Source'
        verbose_name_plural = 'Shellfish Sources'
        ordering = ['-source_date']

    def __str__(self):
        return f"{self.shellfish_species.name} - {self.source_name}"

class StorageCondition(models.Model):
    shellfish_species = models.ForeignKey(Shellfish, related_name='storage_conditions', on_delete=models.CASCADE)
    shellfish_source = models.ForeignKey(
        ShellfishSource, 
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
        source_name = self.shellfish_source.source_name if self.shellfish_source else 'Unknown'
        return f"{self.shellfish_species.name} ({source_name}) at {self.temperature}°C"

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