# Food Safety and Provenance Verification (FoodSPV)

![Django](https://img.shields.io/badge/django-5.1.3-green.svg)
![Python](https://img.shields.io/badge/python-3.x-blue.svg)

A comprehensive Django web application for food safety verification and traceability management, specifically designed for aquatic products including fish, shrimp, and shellfish.

## 🎯 Features

### Core Functionality
- **Multi-Species Management**: Support for fish, shrimp, and shellfish species tracking
- **Source Traceability**: Complete provenance tracking from source to consumer
- **Storage Condition Monitoring**: Environmental parameter tracking and validation
- **Data Analytics**: Statistical analysis and visualization of quality metrics
- **REST API**: RESTful API endpoints for data integration

### Specialized Modules
- **Fish Management** (`fish_app`): Comprehensive fish species database with quality metrics
- **Shrimp Tracking** (`shrimp_app`): Specialized shrimp quality assessment system
- **Shellfish Safety** (`shellfish_app`): Shellfish safety monitoring and verification
- **Main Dashboard** (`main_app`): Central management interface and analytics

## 🏗️ System Architecture

```
FoodSPV/
├── fish_app/           # Fish species management
├── shrimp_app/         # Shrimp quality tracking  
├── shellfish_app/      # Shellfish safety monitoring
├── main_app/           # Dashboard and main interface
├── foodspv_project/    # Django project configuration
├── static/             # Static assets
├── staticfiles/        # Collected static files
├── media/              # User uploaded files
└── manage.py           # Django management script
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- SQLite (for development) or PostgreSQL (for production)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FoodSPV
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv foodspv_env
   # Windows
   foodspv_env\Scripts\activate
   # Linux/Mac
   source foodspv_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Database setup**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create superuser (optional)**
   ```bash
   python manage.py createsuperuser
   ```

6. **Run the development server**
   ```bash
   python manage.py runserver
   ```

7. **Access the application**
   - Web interface: http://127.0.0.1:8000/
   - Admin panel: http://127.0.0.1:8000/admin/

## 📊 Data Models

### Fish Management
- **Fish Species**: Name, scientific classification, descriptions
- **Fish Source**: Traceability data, source types, collection dates
- **Storage Conditions**: Temperature, humidity, storage duration
- **Quality Metrics**: Freshness indicators, safety parameters

### Shrimp Tracking
- **Shrimp Species**: Species information and characteristics
- **Quality Assessment**: Size, color, texture analysis
- **Processing Data**: Processing methods, timestamps
- **Safety Parameters**: Microbial tests, chemical analysis

### Shellfish Safety
- **Shellfish Types**: Species classification and properties
- **Harvest Data**: Location, date, environmental conditions
- **Safety Tests**: Biotoxin levels, bacterial contamination
- **Certification**: Safety certificates and compliance records

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:
```
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost
DATABASE_URL=sqlite:///db.sqlite3
```

### Database Configuration
The default configuration uses SQLite for development. For production, update `settings.py` to use PostgreSQL or MySQL.

## 📚 API Documentation

### Available Endpoints
- `/api/fish/` - Fish species management
- `/api/shrimp/` - Shrimp data operations  
- `/api/shellfish/` - Shellfish tracking
- `/api/sources/` - Source traceability data
- `/api/storage/` - Storage condition records

### Authentication
The API uses Django REST Framework's built-in authentication. Include authentication headers in your requests.

## 🧪 Testing

Run the test suite:
```bash
python manage.py test
```

Run specific app tests:
```bash
python manage.py test fish_app
python manage.py test shrimp_app
python manage.py test shellfish_app
```

## 📈 Data Analytics

The system includes built-in analytics capabilities:
- Quality trend analysis
- Source performance metrics  
- Storage condition optimization
- Safety compliance reporting

## 🔒 Security Features

- User authentication and authorization
- Data validation and sanitization
- CSRF protection
- SQL injection prevention
- Input validation middleware

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Support

For support and questions:
- Create an issue in the GitHub repository
- Contact the development team
- Check the documentation wiki

## 🔄 Version History

- **v1.0.0** - Initial release with basic functionality
- **v1.1.0** - Added API endpoints and authentication
- **v1.2.0** - Enhanced analytics and reporting features

## 🙏 Acknowledgments

- Django community for the excellent framework
- Contributors and testers
- Research institutions providing domain expertise

---

**Note**: This system is designed for research and educational purposes. For production use in commercial food safety applications, please ensure compliance with local regulations and standards.