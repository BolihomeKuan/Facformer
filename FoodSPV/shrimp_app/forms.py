from django import forms
from .models import Shrimp, ShrimpSource, StorageCondition, FeatureType, FeatureData

class ShrimpForm(forms.ModelForm):
    class Meta:
        model = Shrimp
        fields = ['name', 'scientific_name', 'description', 'image']
        widgets = {
            'description': forms.Textarea(attrs={'rows': 4}),
        }

class ShrimpSourceForm(forms.ModelForm):
    class Meta:
        model = ShrimpSource
        fields = ['source_code', 'source_type', 'source_name', 
                 'source_date', 'description']
        widgets = {
            'source_date': forms.DateInput(attrs={'type': 'date'}),
            'description': forms.Textarea(attrs={'rows': 3}),
        }

class StorageConditionForm(forms.ModelForm):
    class Meta:
        model = StorageCondition
        fields = ['temperature', 'storage_time', 'notes']
        widgets = {
            'notes': forms.Textarea(attrs={'rows': 3}),
        }

class FeatureDataForm(forms.ModelForm):
    feature_type = forms.CharField(
        max_length=100,
        label='Feature Type',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'example: pH, TVB-N etc.'
        })
    )
    
    class Meta:
        model = FeatureData
        fields = ['value', 'measurement_day', 'notes']
        widgets = {
            'value': forms.NumberInput(attrs={'class': 'form-control'}),
            'measurement_day': forms.NumberInput(attrs={'class': 'form-control'}),
            'notes': forms.Textarea(attrs={'rows': 2, 'class': 'form-control'})
        }

class DataImportForm(forms.Form):
    file = forms.FileField(
        label='Select File',
        help_text='Supported formats: CSV, Excel'
    )
    sheet_name = forms.CharField(
        required=False,
        help_text='Leave blank for CSV files'
    )


class BulkFeatureDataForm(forms.Form):
    bulk_data = forms.CharField(
        widget=forms.Textarea(
            attrs={
                'rows': 10,
                'class': 'form-control',
                'placeholder': 'Enter data in format:\nFeature_Type Day Value Note(optional)\nExample:\nTVB-N 1 2.5 Note1\nTVB-N 2 3.1\nTVB-N 3 4.2 Some note'
            }
        ),
        help_text='Enter one measurement per line. Format: Feature_Type Day Value Note(optional)'
    )