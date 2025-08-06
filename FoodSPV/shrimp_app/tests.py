from django.test import TestCase, Client
from django.urls import reverse
from .models import Shrimp, StorageCondition, FeatureType, FeatureData
from django.contrib.auth.models import User

class ShrimpAppTests(TestCase):
    def setUp(self):
        # 创建测试用户
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.client = Client()
        
        # 创建测试数据
        self.shrimp = Shrimp.objects.create(
            name='Test Shrimp',
            scientific_name='Test Scientific Name',
            description='Test Description'
        )
        
    def test_index_view(self):
        response = self.client.get(reverse('shrimp_app:index'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'shrimp_app/index.html')
    
    def test_add_shrimp(self):
        self.client.login(username='testuser', password='testpass123')
        response = self.client.post(reverse('shrimp_app:add_shrimp'), {
            'name': 'New Shrimp',
            'scientific_name': 'New Scientific Name',
            'description': 'New Description'
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(Shrimp.objects.filter(name='New Shrimp').exists())
