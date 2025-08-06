from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from fish_app.models import FishSpecies, StorageCondition, FeatureType, FeatureData

class SystemIntegrationTest(TestCase):
    def setUp(self):
        # 创建测试用户
        self.user = User.objects.create_superuser(
            username='admin',
            email='admin@example.com',
            password='admin123'
        )
        self.client = Client()
        self.client.login(username='admin', password='admin123')
        
        # 创建基础数据
        self.fish = FishSpecies.objects.create(
            name='Test Fish',
            scientific_name='Test Scientific Name',
            description='Test Description'
        )
        
        self.feature = FeatureType.objects.create(
            name='Test Feature',
            description='Test Feature Description',
            unit='test unit'
        )
        
        self.storage = StorageCondition.objects.create(
            fish_species=self.fish,
            temperature=4.0,
            storage_time=14,
            notes='Test storage'
        )

    def test_complete_workflow(self):
        """测试完整工作流程"""
        # 测试首页访问
        response = self.client.get(reverse('fish_app:index'))
        self.assertEqual(response.status_code, 200)
        
        # 测试添加数据
        response = self.client.post(reverse('fish_app:add_feature_data', args=[self.storage.id]), {
            'feature_type': self.feature.id,
            'value': 10.5,
            'measurement_day': 0
        })
        self.assertEqual(response.status_code, 302)
        
        # 测试数据查看
        response = self.client.get(reverse('fish_app:fish_detail', args=[self.fish.id]))
        self.assertEqual(response.status_code, 200)
        
        # 测试数据编辑
        response = self.client.post(reverse('fish_app:edit_fish', args=[self.fish.id]), {
            'name': 'Updated Fish',
            'scientific_name': self.fish.scientific_name,
            'description': self.fish.description
        })
        self.assertEqual(response.status_code, 302)
        
        # 测试数据导出
        response = self.client.get(reverse('fish_app:export_all_data', args=[self.fish.id]))
        self.assertEqual(response.status_code, 200)
        
        # 测试统计分析
        response = self.client.get(reverse('fish_app:statistical_analysis'))
        self.assertEqual(response.status_code, 200) 