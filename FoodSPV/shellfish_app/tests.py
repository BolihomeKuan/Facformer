from django.test import TestCase, Client
from django.urls import reverse
from .models import ShellfishSpecies, StorageCondition, FeatureType, FeatureData
from django.contrib.auth.models import User

class ShellfishAppTests(TestCase):
    def setUp(self):
        # 创建测试用户
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123'
        )
        self.client = Client()
        
        # 创建测试数据
        self.shellfish = ShellfishSpecies.objects.create(
            name='Test Shellfish',
            scientific_name='Test Scientific Name',
            description='Test Description'
        )
        
    def test_index_view(self):
        response = self.client.get(reverse('shellfish_app:index'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'shellfish_app/index.html')
    
    def test_add_shellfish(self):
        self.client.login(username='testuser', password='testpass123')
        response = self.client.post(reverse('shellfish_app:add_shellfish'), {
            'name': 'New Shellfish',
            'scientific_name': 'New Scientific Name',
            'description': 'New Description'
        })
        self.assertEqual(response.status_code, 302)
        self.assertTrue(ShellfishSpecies.objects.filter(name='New Shellfish').exists())
