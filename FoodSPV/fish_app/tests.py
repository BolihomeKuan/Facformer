from django.test import TestCase, Client
from django.urls import reverse
from .models import FishSpecies, StorageCondition, FeatureType, FeatureData
from django.contrib.auth.models import User

class FishAppTests(TestCase):
    def setUp(self):
        # 创建测试用户
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.client = Client()
        
        # 创建测试数据
        self.fish = FishSpecies.objects.create(
            name='Test Fish',
            scientific_name='Test Scientific Name',
            description='Test Description'
        )
        
    def test_index_view(self):
        response = self.client.get(reverse('fish_app:index'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'fish_app/index.html')
    
    def test_add_fish(self):
        self.client.login(username='testuser', password='testpass123')
        response = self.client.post(reverse('fish_app:add_fish'), {
            'name': 'New Fish',
            'scientific_name': 'New Scientific Name',
            'description': 'New Description'
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(FishSpecies.objects.filter(name='New Fish').exists())
