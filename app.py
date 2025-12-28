import logging
import os
import shutil
import tempfile
import time
import traceback

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

# Настройка логгера
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

# Импорт TSR компонентов
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
from tsr.bake_texture import bake_texture
import rembg
import xatlas


# ================ ФУНКЦИЯ ИНИЦИАЛИЗАЦИИ МОДЕЛИ ================
@st.cache_resource
def load_tsr_model(device="cuda:0", chunk_size=8192, mc_resolution=256):
    """Загрузка предобученной модели TSR с кэшированием"""
    st.info("🧠 Загрузка модели TripoSR...")

    try:
        # Проверка устройства
        if not torch.cuda.is_available():
            device = "cpu"
            st.warning("⚠️ CUDA недоступна. Используется CPU (работа будет медленнее)")

        # Загрузка модели
        model = TSR.from_pretrained(
            "stabilityai/TripoSR",
            config_name="config.yaml",
            weight_name="model.ckpt",
        )

        # Настройка параметров
        model.renderer.set_chunk_size(chunk_size)
        model.to(device)

        st.success(f"✅ Модель успешно загружена на {device}!")
        return model, device

    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {str(e)}")
        st.code(traceback.format_exc())
        raise


# ================ ФУНКЦИЯ ОБРАБОТКИ ИЗОБРАЖЕНИЯ ================
def process_image(image, remove_bg=True, foreground_ratio=0.85):
    """Предобработка изображения перед генерацией"""
    if remove_bg:
        st.info("🧹 Удаление фона...")
        session = rembg.new_session()
        image = remove_background(image, session)
        image = resize_foreground(image, foreground_ratio)

        # Конвертация в правильный формат
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))

    return image


# ================ ФУНКЦИЯ ГЕНЕРАЦИИ 3D МОДЕЛИ ================
def generate_full_3d(model, image, device, output_dir, bake=True, texture_res=2048, mc_resolution=256, format="obj"):
    """
    Полная генерация 3D модели с сохранением UV координат для визуализации
    """
    try:
        # 1. Генерация scene_codes
        with torch.no_grad():
            scene_codes = model([image], device=device)

        # 2. Извлечение mesh
        meshes = model.extract_mesh(scene_codes, has_vertex_color=True, resolution=mc_resolution)
        mesh = meshes[0]

        output_files = {}

        if bake:
            # 3. БАКИНГ ТЕКСТУРЫ И ЭКСПОРТ
            st.info("🎨 Выпечка текстуры...")
            bake_output = bake_texture(mesh, model, scene_codes[0], texture_res)

            # Пути для сохранения
            mesh_path = os.path.join(output_dir, f"model.{format}")
            texture_path = os.path.join(output_dir, "texture.png")

            # Экспорт через xatlas
            xatlas.export(
                mesh_path,
                mesh.vertices[bake_output["vmapping"]],
                bake_output["indices"],
                bake_output["uvs"],
                mesh.vertex_normals[bake_output["vmapping"]]
            )

            # Сохранение текстуры
            texture_img = Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(
                Image.FLIP_TOP_BOTTOM)
            texture_img.save(texture_path)

            # СОХРАНЯЕМ UV КООРДИНАТЫ ДЛЯ ВИЗУАЛИЗАЦИИ
            output_files['vertices'] = mesh.vertices[bake_output["vmapping"]]
            output_files['indices'] = bake_output["indices"]
            output_files['uvs'] = bake_output["uvs"]  # UV координаты для визуализации

            output_files['mesh'] = mesh_path
            output_files['texture'] = texture_path
            output_files['has_texture'] = True
            output_files['texture_image'] = texture_img  # Сохраняем изображение для визуализации

        else:
            # 4. ЭКСПОРТ БЕЗ ТЕКСТУРЫ
            mesh_path = os.path.join(output_dir, f"model.{format}")
            mesh.export(mesh_path)

            output_files['mesh'] = mesh_path
            output_files['has_texture'] = False
            output_files['vertices'] = mesh.vertices
            output_files['faces'] = mesh.faces
            if hasattr(mesh, 'vertex_colors'):
                output_files['colors'] = mesh.vertex_colors

        return output_files

    except Exception as e:
        st.error(f"❌ Ошибка в процессе генерации: {str(e)}")
        st.code(traceback.format_exc())
        return None


# ================ ВИЗУАЛИЗАЦИЯ С НАСТОЯЩЕЙ ТЕКСТУРОЙ ================
def visualize_3d_model_with_real_texture(mesh_data, texture_image=None):
    """Визуализация 3D модели с настоящим UV-маппингом текстуры"""
    try:
        if not mesh_data.get('has_texture', False) or texture_image is None:
            return visualize_3d_model_standard(mesh_data)

        st.info("🎨 Применение настоящей текстуры через UV-маппинг...")

        verts = mesh_data['vertices']
        faces = mesh_data['indices']
        uvs = mesh_data['uvs']  # UV координаты из bake_texture

        # Конвертируем текстуру в numpy массив
        texture_array = np.array(texture_image)
        tex_height, tex_width = texture_array.shape[0], texture_array.shape[1]

        # Создаем массив цветов для вершин (по UV координатам)
        vertex_colors = np.zeros((len(verts), 3))

        # Маппим UV координаты на цвета текстуры
        for i, uv in enumerate(uvs):
            if i >= len(verts):
                break

            # UV координаты обычно в диапазоне [0, 1]
            u = max(0, min(1, uv[0]))
            v = max(0, min(1, uv[1]))

            # Конвертируем UV в пиксельные координаты (с инверсией Y)
            x = int(u * (tex_width - 1))
            y = int((1 - v) * (tex_height - 1))  # Инвертируем Y для правильного отображения

            # Берем цвет пикселя
            if y < tex_height and x < tex_width:
                color = texture_array[y, x, :3]  # Берем RGB каналы
                vertex_colors[i] = color / 255.0  # Нормализуем в [0, 1]

        # Нормализуем цвета (на случай выхода за пределы)
        vertex_colors = np.clip(vertex_colors, 0, 1)

        # Создаем визуализацию с вершинными цветами
        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=vertex_colors,  # НАСТОЯЩИЕ ЦВЕТА ВЕРШИН
                flatshading=True,
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.7,
                    specular=0.3,
                    roughness=0.2
                ),
                lightposition=dict(x=100, y=200, z=0)
            )
        ])

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=500,
            title="3D модель с настоящей текстурой (UV-mapping)"
        )

        # Показываем сравнение: текстура vs 3D модель
        col_tex1, col_tex2 = st.columns([1, 1])
        with col_tex1:
            st.subheader("🖼️ Исходная текстура")
            st.image(texture_image, use_column_width=True)
            st.write(f"Размер: {tex_width}×{tex_height} пикселей")

        with col_tex2:
            st.subheader("📊 Статистика UV-маппинга")
            valid_colors = np.sum(np.any(vertex_colors > 0, axis=1))
            st.write(f"Вершины с текстурой: {valid_colors}/{len(verts)}")
            st.write(f"UV диапазон: U[0-{np.max(uvs[:, 0]):.2f}], V[0-{np.max(uvs[:, 1]):.2f}]")

        return fig

    except Exception as e:
        st.error(f"❌ Ошибка UV-маппинга: {str(e)}")
        st.code(traceback.format_exc())
        # Fallback на стандартную визуализацию
        return visualize_3d_model_standard(mesh_data)


def visualize_3d_model_standard(mesh_data):
    """Стандартная визуализация (без текстуры)"""
    try:
        if mesh_data.get('has_texture', False):
            verts = mesh_data['vertices']
            faces = mesh_data['indices']
            i_faces = faces[:, 0] if len(faces.shape) > 1 else faces[0::3]
            j_faces = faces[:, 1] if len(faces.shape) > 1 else faces[1::3]
            k_faces = faces[:, 2] if len(faces.shape) > 1 else faces[2::3]
            color_hex = '#8B7355'  # Бежевый
        else:
            verts = mesh_data['vertices']
            faces = mesh_data['faces']
            i_faces = faces[:, 0]
            j_faces = faces[:, 1]
            k_faces = faces[:, 2]

            # Используем вершинные цвета если есть
            colors = mesh_data.get('colors')
            if colors is not None and len(colors) > 0:
                avg_color = np.mean(colors, axis=0)
                if np.max(avg_color) > 1.0:
                    avg_color = avg_color / 255.0
                color_hex = f'#{int(avg_color[0] * 255):02x}{int(avg_color[1] * 255):02x}{int(avg_color[2] * 255):02x}'
            else:
                color_hex = '#8B4513'  # Коричневый

        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=i_faces,
                j=j_faces,
                k=k_faces,
                color=color_hex,
                opacity=0.9,
                flatshading=True,
                lighting=dict(
                    ambient=0.3,
                    diffuse=0.8,
                    specular=0.1,
                    roughness=0.5
                ),
                lightposition=dict(x=100, y=200, z=0)
            )
        ])

        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=500
        )

        return fig

    except Exception as e:
        st.error(f"❌ Ошибка стандартной визуализации: {str(e)}")
        return go.Figure()


# ================ ОСНОВНОЕ ПРИЛОЖЕНИЕ ================
def main():
    st.set_page_config(
        page_title="🎨 TripoSR 3D Generator",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎨 TripoSR 3D Generator")
    st.markdown("### Генерация 3D моделей из 2D изображений с помощью Stability AI")

    # Инициализация session_state
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.device = None
        st.session_state.model_loaded = False
        st.session_state.generated_files = None
        st.session_state.processed_image = None
        st.session_state.texture_image = None

    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки генерации")

        # Кнопка загрузки модели
        if not st.session_state.model_loaded:
            if st.button("🚀 Загрузить модель TripoSR", type="primary"):
                with st.spinner("Загрузка модели (это займет время при первом запуске)..."):
                    try:
                        model, device = load_tsr_model()
                        st.session_state.model = model
                        st.session_state.device = device
                        st.session_state.model_loaded = True
                        st.success("✅ Модель загружена!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Не удалось загрузить модель: {e}")

        if st.session_state.model_loaded:
            st.success(f"✅ Модель загружена на {st.session_state.device}")

            # Настройки генерации
            st.subheader("Параметры обработки")
            remove_bg = st.checkbox("Автоматически удалить фон", value=True, help="Использует rembg для удаления фона")
            if remove_bg:
                foreground_ratio = st.slider("Размер объекта на изображении", 0.5, 1.0, 0.85, 0.05)

            st.subheader("Параметры 3D модели")
            bake_texture = st.checkbox("Выпечь текстуру", value=True,
                                       help="Создает текстуру atlas вместо vertex colors")
            if bake_texture:
                texture_resolution = st.select_slider(
                    "Разрешение текстуры",
                    options=[512, 1024, 2048, 4096],
                    value=2048
                )

            mc_resolution = st.select_slider(
                "Качество сетки (Marching Cubes resolution)",
                options=[128, 256, 512],
                value=256,
                help="Выше значение = детальнее модель, но дольше генерация"
            )

            output_format = st.radio(
                "Формат экспорта",
                options=["obj", "glb"],
                index=0,
                horizontal=True
            )

            st.divider()
            st.subheader("ℹ️ Информация")
            st.info(f"""
            **Устройство:** {st.session_state.device.upper()}
            **Качество сетки:** {mc_resolution}
            **Формат:** {output_format.upper()}
            **Примерное время генерации:** 30 сек - 2 мин
            """)

    # Основной контент
    if not st.session_state.model_loaded:
        st.warning("⚠️ Модель не загружена. Нажмите кнопку в боковой панели.")
        st.info("""
        **Первая загрузка модели может занять несколько минут**, так как будут скачаны веса (~2.5GB).
        Убедитесь, что у вас стабильное интернет-соединение.
        """)
        return

    # Загрузка изображения
    st.header("🖼️ Загрузите изображение")
    uploaded_file = st.file_uploader(
        "Выберите файл (PNG, JPG, JPEG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        # Показываем исходное изображение
        original_image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📸 Исходное изображение")
            st.image(original_image, use_column_width=True)

            # Кнопка предварительной обработки
            if st.button("✂️ Обработать изображение", type="secondary"):
                with st.spinner("Обработка..."):
                    processed_image = process_image(
                        original_image.copy(),
                        remove_bg=remove_bg,
                        foreground_ratio=foreground_ratio
                    )
                    st.session_state.processed_image = processed_image
                    st.success("Готово!")
                    st.rerun()

        # Показываем обработанное изображение, если есть
        if 'processed_image' in st.session_state and st.session_state.processed_image is not None:
            with col2:
                st.subheader("✨ После обработки")
                st.image(st.session_state.processed_image, use_column_width=True)

                # Основная кнопка генерации
                if st.button("🚀 СГЕНЕРИРОВАТЬ 3D МОДЕЛЬ", type="primary", use_container_width=True):
                    with st.spinner("⏳ Генерация 3D модели... Это может занять от 30 секунд до 2 минут"):
                        try:
                            start_time = time.time()

                            # Создаем временную директорию для результатов
                            with tempfile.TemporaryDirectory() as tmpdir:
                                # Генерация модели
                                result = generate_full_3d(
                                    st.session_state.model,
                                    st.session_state.processed_image,
                                    st.session_state.device,
                                    output_dir=tmpdir,
                                    bake=bake_texture,
                                    texture_res=texture_resolution,
                                    mc_resolution=mc_resolution,
                                    format=output_format
                                )

                                if result:
                                    generation_time = time.time() - start_time
                                    st.session_state.generated_files = result
                                    st.session_state.generation_time = generation_time

                                    # Сохраняем текстуру в session_state для визуализации
                                    if bake_texture and 'texture_image' in result:
                                        st.session_state.texture_image = result['texture_image']
                                        st.success("✅ Текстура готова для визуализации!")

                                    # Сохраняем файлы в постоянное место для скачивания
                                    os.makedirs("output", exist_ok=True)
                                    timestamp = int(time.time())
                                    perm_dir = f"output/generation_{timestamp}"
                                    os.makedirs(perm_dir, exist_ok=True)

                                    # Копируем файлы
                                    for key in ['mesh', 'texture']:
                                        if key in result and result[key]:
                                            src = result[key]
                                            dst = os.path.join(perm_dir, os.path.basename(src))
                                            shutil.copy2(src, dst)
                                            result[f'{key}_perm'] = dst

                                    st.session_state.generated_files = result
                                    st.success(f"✅ Модель сгенерирована за {generation_time:.1f} секунд!")
                                    st.rerun()

                        except Exception as e:
                            st.error(f"❌ Ошибка при генерации: {str(e)}")
                            st.code(traceback.format_exc())

        # Показываем результат, если есть
        if 'generated_files' in st.session_state and st.session_state.generated_files:
            st.divider()
            st.header("🎮 Результат 3D генерации")

            # Визуализация
            if st.session_state.generated_files.get('has_texture') and hasattr(st.session_state,
                                                                               'texture_image') and st.session_state.texture_image is not None:
                fig = visualize_3d_model_with_real_texture(
                    st.session_state.generated_files,
                    st.session_state.texture_image
                )
            else:
                fig = visualize_3d_model_standard(st.session_state.generated_files)

            st.plotly_chart(fig, use_container_width=True)

            # Информация
            col_info1, col_info2 = st.columns([1, 1])
            with col_info1:
                st.metric("Время генерации", f"{st.session_state.generation_time:.1f} сек")
                verts = st.session_state.generated_files.get('vertices')
                if verts is not None:
                    st.metric("Количество вершин", len(verts))

            with col_info2:
                if st.session_state.generated_files.get('has_texture'):
                    st.metric("Текстура", "Да", f"{texture_resolution}px")
                else:
                    st.metric("Текстура", "Нет (vertex colors)")

            # Секция скачивания
            st.header("📥 Экспорт модели")
            download_col1, download_col2 = st.columns([1, 1])

            mesh_path = st.session_state.generated_files.get('mesh_perm')
            texture_path = st.session_state.generated_files.get('texture_perm')

            if mesh_path and os.path.exists(mesh_path):
                with open(mesh_path, 'rb') as f:
                    mesh_bytes = f.read()

                with download_col1:
                    st.download_button(
                        label=f"📥 Скачать {output_format.upper()} модель",
                        data=mesh_bytes,
                        file_name=f"triposr_model.{output_format}",
                        mime="application/octet-stream",
                        use_container_width=True
                    )

            if texture_path and os.path.exists(texture_path):
                with open(texture_path, 'rb') as f:
                    texture_bytes = f.read()

                with download_col2:
                    st.download_button(
                        label="🎨 Скачать текстуру (PNG)",
                        data=texture_bytes,
                        file_name="triposr_texture.png",
                        mime="image/png",
                        use_container_width=True
                    )

            # Кнопка очистки
            if st.button("🔄 Начать заново", type="secondary"):
                keys_to_clear = ['processed_image', 'generated_files', 'generation_time', 'texture_image']
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()


if __name__ == "__main__":
    main()
