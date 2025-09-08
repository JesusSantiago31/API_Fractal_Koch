import numpy as np
import matplotlib
# Usar backend Agg que no requiere GUI - evita errores de Tkinter en servidores
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import io
import base64
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for
from flask_cors import CORS

# Crear aplicación Flask principal
app = Flask(__name__)
# Habilitar CORS para permitir requests desde diferentes dominios
CORS(app)

# Configuración de la carpeta donde se guardarán las imágenes
UPLOAD_FOLDER = 'static/images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Crear carpeta si no existe

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Configurar la carpeta en la app

def koch_snowflake_external_step(points, step, scale_factor=1.0):
    """
    Genera un paso del algoritmo del copo de nieve de Koch con triángulos externos.
    
    Args:
        points: Array de puntos actuales del fractal
        step: Número de iteración actual (0 = triángulo inicial)
        scale_factor: Factor de escala para el tamaño
    
    Returns:
        np.array: Nuevos puntos después de aplicar la transformación de Koch
    """
    if step == 0:
        # Paso inicial: crear un triángulo equilátero
        side_length = 1.0 * scale_factor
        height = (np.sqrt(3) / 2) * side_length  # Altura de triángulo equilátero
        
        # Definir los 3 puntos del triángulo
        p1 = np.array([0, 0])
        p2 = np.array([side_length, 0])
        p3 = np.array([side_length / 2, height])
        
        return np.array([p1, p2, p3, p1])  # Devolver puntos + cerrar el triángulo
    
    new_points = []  # Lista para almacenar los nuevos puntos
    
    # Aplicar la transformación de Koch a cada segmento
    for i in range(len(points) - 1):
        p0 = points[i]          # Punto inicial del segmento
        p1_val = points[i + 1]  # Punto final del segmento
        
        # Dividir el segmento en 3 partes iguales
        segment = p1_val - p0
        one_third = p0 + segment / 3    # Punto a 1/3 del segmento
        two_thirds = p0 + 2 * segment / 3  # Punto a 2/3 del segmento
        
        # Matriz de rotación para -60 grados (triángulos externos)
        rotation_matrix = np.array([[np.cos(-np.pi/3), -np.sin(-np.pi/3)],
                                   [np.sin(-np.pi/3), np.cos(-np.pi/3)]])
        
        # Vector desde one_third a two_thirds
        vec = two_thirds - one_third
        # Rotar el vector para crear el pico del triángulo
        rotated_vec = np.dot(rotation_matrix, vec)
        # Calcular la posición del pico
        peak = one_third + rotated_vec
        
        # Añadir todos los puntos del nuevo segmento transformado
        new_points.extend([p0, one_third, peak, two_thirds])
    
    new_points.append(points[-1])  # Añadir el último punto para cerrar la figura
    return np.array(new_points)    # Convertir a array numpy

def generate_koch_snowflake(iterations=4, scale=2.0, half_type='complete'):
    """
    Genera el copo de nieve completo o sus mitades mediante iteraciones sucesivas.
    
    Args:
        iterations: Número de iteraciones del fractal (0-8)
        scale: Factor de escala para el tamaño
        half_type: Tipo de copo a generar ('complete', 'inferior', 'superior', 'izquierda', 'derecha')
    
    Returns:
        np.array: Puntos que forman el copo de nieve o la mitad seleccionada
    """
    # Generar copo completo primero mediante iteraciones sucesivas
    steps = []  # Almacenar cada paso de la iteración
    current_points = koch_snowflake_external_step(None, 0, scale)  # Triángulo inicial
    steps.append(current_points)
    
    # Aplicar las iteraciones solicitadas
    for i in range(1, iterations + 1):
        current_points = koch_snowflake_external_step(steps[-1], i, scale)
        steps.append(current_points)
    
    points_complete = steps[-1]  # Obtener el resultado final completo
    
    # Filtrar puntos según el tipo de mitad solicitada
    if half_type == 'complete':
        return points_complete  # Devolver copo completo
    elif half_type == 'inferior':
        # Mitad inferior: puntos con y <= mitad de la altura
        mid_y = np.max(points_complete[:, 1]) / 2
        return points_complete[points_complete[:, 1] <= mid_y]
    elif half_type == 'superior':
        # Mitad superior: puntos con y >= mitad de la altura
        mid_y = np.max(points_complete[:, 1]) / 2
        return points_complete[points_complete[:, 1] >= mid_y]
    elif half_type == 'izquierda':
        # Mitad izquierda: puntos con x <= mitad del ancho
        mid_x = np.max(points_complete[:, 0]) / 2
        return points_complete[points_complete[:, 0] <= mid_x]
    elif half_type == 'derecha':
        # Mitad derecha: puntos con x >= mitad del ancho
        mid_x = np.max(points_complete[:, 0]) / 2
        return points_complete[points_complete[:, 0] >= mid_x]
    else:
        return points_complete  # Por defecto, devolver completo

def create_koch_image(points, scale, iterations, color='blue', half_type='complete', filename=None):
    """
    Crea y guarda/muestra la imagen del copo de nieve usando matplotlib.
    
    Args:
        points: Array de puntos a graficar
        scale: Factor de escala usado
        iterations: Número de iteraciones
        color: Color de las líneas
        half_type: Tipo de copo generado
        filename: Si se proporciona, guarda la imagen en este archivo
    
    Returns:
        str: Nombre del archivo si se guardó, o string base64 si no
    """
    # Desactivar modo interactivo para evitar problemas en servidor
    plt.ioff()  
    # Crear figura y eje para el gráfico
    fig, ax = plt.subplots(figsize=(10, 10))
    
    try:
        # Dibujar el copo de nieve como línea continua
        ax.plot(points[:, 0], points[:, 1], color, linewidth=1.5)
        
        # Solo rellenar el área si es el copo completo
        if half_type == 'complete':
            ax.fill(points[:, 0], points[:, 1], 'lightblue', alpha=0.3)
        
        # Configurar aspecto y quitar ejes
        ax.set_aspect('equal')  # Mantener proporciones correctas
        ax.axis('off')          # Ocultar ejes y bordes
        
        # Añadir título descriptivo
        title = f'Copo de Koch - {half_type.capitalize()} - {iterations} iteraciones'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Ajustar límites de visualización según el tipo de copo
        if half_type == 'complete':
            margin = 0.5 * scale
            ax.set_xlim(-margin, np.max(points[:, 0]) + margin)
            ax.set_ylim(-margin, np.max(points[:, 1]) + margin)
        elif half_type == 'inferior':
            ax.set_xlim(-0.5 * scale, np.max(points[:, 0]) + 0.5 * scale)
            ax.set_ylim(-0.1 * scale, np.max(points[:, 1]) + 0.1 * scale)
        elif half_type == 'superior':
            ax.set_xlim(-0.5 * scale, np.max(points[:, 0]) + 0.5 * scale)
            ax.set_ylim(np.min(points[:, 1]) - 0.1 * scale, np.max(points[:, 1]) + 0.1 * scale)
        elif half_type == 'izquierda':
            ax.set_xlim(-0.1 * scale, np.max(points[:, 0]) + 0.1 * scale)
            ax.set_ylim(-0.5 * scale, np.max(points[:, 1]) + 0.5 * scale)
        elif half_type == 'derecha':
            ax.set_xlim(np.min(points[:, 0]) - 0.1 * scale, np.max(points[:, 0]) + 0.1 * scale)
            ax.set_ylim(-0.5 * scale, np.max(points[:, 1]) + 0.5 * scale)
        
        # Guardar imagen en archivo si se proporciona filename
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white', format='png')
            plt.close(fig)
            return filename  # Devolver nombre del archivo
        else:
            # Convertir imagen a base64 para respuesta API
            img_buffer = io.BytesIO()  # Buffer en memoria
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
            img_buffer.seek(0)  # Volver al inicio del buffer
            # Codificar imagen en base64
            img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            return img_data  # Devolver string base64
            
    finally:
        # Bloque finally asegura que las figuras se cierren incluso con errores
        plt.close('all')  # Cerrar todas las figuras
        plt.ion()         # Reactivar modo interactivo

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Ruta principal que maneja tanto GET (mostrar formulario) como POST (procesar formulario).
    """
    if request.method == 'POST':
        try:
            # Verificar si es una solicitud de borrado de imágenes
            if 'clear_images' in request.form:
                clear_images()  # Borrar todas las imágenes
                return redirect(url_for('index'))  # Recargar página
            
            # Obtener parámetros del formulario HTML
            level = int(request.form.get('level', 3))
            scale = float(request.form.get('scale', 2.0))
            color = request.form.get('color', 'blue')
            half_type = request.form.get('half_type', 'complete')
            
            # Validar parámetros recibidos
            if level < 0 or level > 8:
                return render_template('index.html', error="Las iteraciones deben estar entre 0 y 8")
            
            if scale <= 0 or scale > 10:
                return render_template('index.html', error="La escala debe estar entre 0.1 y 10")
            
            # Generar los puntos del copo de nieve
            points = generate_koch_snowflake(level, scale, half_type)
            
            # Crear nombre de archivo único con timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"koch_{half_type}_{level}iter_{scale}scale_{timestamp}.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Crear y guardar la imagen
            create_koch_image(points, scale, level, color, half_type, filepath)
            
            # Obtener lista actualizada de imágenes
            images = get_existing_images()
            
            # Renderizar template con resultados
            return render_template('index.html', 
                                 level=level, 
                                 scale=scale, 
                                 color=color,
                                 half_type=half_type,
                                 images=images,
                                 outdir=app.config['UPLOAD_FOLDER'],
                                 success=True)
            
        except Exception as e:
            # Manejar cualquier error inesperado
            return render_template('index.html', error=f"Error: {str(e)}")
    
    # Método GET: mostrar formulario vacío
    images = get_existing_images()  # Obtener imágenes existentes
    return render_template('index.html', images=images, outdir=app.config['UPLOAD_FOLDER'])

def get_existing_images():
    """
    Obtiene la lista de todas las imágenes PNG en la carpeta de uploads.
    
    Returns:
        list: Lista de URLs relativas a las imágenes
    """
    images = []
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        # Listar archivos ordenados inversamente (más recientes primero)
        for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER']), reverse=True):
            if filename.endswith('.png'):
                # Añadir URL relativa a la imagen
                images.append(f"/static/images/{filename}")
    return images

def clear_images():
    """
    Elimina todas las imágenes PNG de la carpeta de uploads.
    Ignora errores si algún archivo no se puede borrar.
    """
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.endswith('.png'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    os.remove(filepath)  # Intentar borrar archivo
                except:
                    pass  # Ignorar errores silenciosamente

@app.route('/api/koch/generate', methods=['POST', 'GET'])
def generate_koch_api():
    """
    Endpoint API RESTful para generar copos de nieve programáticamente.
    Soporta tanto GET con query parameters como POST con JSON.
    """
    try:
        # Obtener parámetros según el método HTTP
        if request.method == 'POST':
            data = request.get_json() or {}  # JSON en body
            iterations = int(data.get('iterations', 4))
            scale = float(data.get('scale', 2.0))
            color = data.get('color', 'blue')
            half_type = data.get('half_type', 'complete')
            return_image = data.get('return_image', True)
        else:
            # GET con query parameters
            iterations = int(request.args.get('iterations', 4))
            scale = float(request.args.get('scale', 2.0))
            color = request.args.get('color', 'blue')
            half_type = request.args.get('half_type', 'complete')
            return_image = request.args.get('return_image', 'true').lower() == 'true'
        
        # Validar parámetros de la API
        if iterations < 0 or iterations > 8:
            return jsonify({'error': 'Iteraciones deben estar entre 0 y 8'}), 400
        if scale <= 0 or scale > 10:
            return jsonify({'error': 'Escala debe estar entre 0.1 y 10'}), 400
        if half_type not in ['complete', 'inferior', 'superior', 'izquierda', 'derecha']:
            return jsonify({'error': 'Tipo debe ser: complete, inferior, superior, izquierda, derecha'}), 400
        
        # Generar el copo de nieve
        points = generate_koch_snowflake(iterations, scale, half_type)
        
        # Calcular métricas interesantes del fractal
        total_points = len(points)
        total_segments = total_points - 1
        estimated_length = total_segments * (scale / (3 ** iterations))
        
        # Preparar respuesta JSON con metadata
        response_data = {
            'success': True,
            'metadata': {
                'iterations': iterations,
                'scale': scale,
                'color': color,
                'half_type': half_type,
                'total_points': total_points,
                'total_segments': total_segments,
                'estimated_length': round(estimated_length, 4),
                'generated_at': datetime.now().isoformat(),
                'fractal_dimension': round(np.log(4) / np.log(3), 4)  # Dimensión fractal teórica
            }
        }
        
        # Incluir imagen en base64 si se solicita
        if return_image:
            image_base64 = create_koch_image(points, scale, iterations, color, half_type)
            response_data['image_base64'] = image_base64
        
        return jsonify(response_data)  # Devolver JSON response
    
    except Exception as e:
        # Manejar errores de la API
        return jsonify({'error': str(e)}), 500

@app.route('/api/koch/list', methods=['GET'])
def list_images_api():
    """
    Endpoint API para listar todas las imágenes generadas con información detallada.
    """
    try:
        images = []
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for filename in sorted(os.listdir(app.config['UPLOAD_FOLDER']), reverse=True):
                if filename.endswith('.png'):
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file_info = {
                        'filename': filename,
                        'url': f"/static/images/{filename}",
                        'size': os.path.getsize(filepath),  # Tamaño en bytes
                        'created': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                    }
                    images.append(file_info)
        
        return jsonify({'images': images, 'total': len(images)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/images/<filename>')
def serve_image(filename):
    """
    Servir imágenes estáticas directamente desde la carpeta.
    
    Args:
        filename: Nombre del archivo de imagen a servir
    
    Returns:
        File: La imagen solicitada o error 404 si no existe
    """
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath)  # Servir archivo
    else:
        return "Imagen no encontrada", 404  # Error HTTP 404

@app.route('/api/koch/clear', methods=['POST'])
def clear_images_api():
    """
    Endpoint API para borrar todas las imágenes mediante programación.
    """
    try:
        clear_images()  # Llamar a la función de borrado
        return jsonify({'success': True, 'message': 'Todas las imágenes han sido eliminadas'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Manejador de errores 404 - Página no encontrada"""
    return render_template('index.html', error="Página no encontrada"), 404

@app.errorhandler(500)
def internal_error(error):
    """Manejador de errores 500 - Error interno del servidor"""
    return render_template('index.html', error="Error interno del servidor"), 500

if __name__ == '__main__':
    # Mensajes de inicio
    print("🚀 Iniciando Koch Snowflake Generator...")
    print("🌐 Servidor web disponible en: http://localhost:5000")
    print("🔧 Backend: Matplotlib con Agg (sin GUI)")
    print("✅ Error de Tkinter solucionado")
    
    # Ejecutar servidor Flask
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)