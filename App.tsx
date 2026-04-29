import {
  View,
  Text,
  TouchableOpacity,
  Image,
  StyleSheet,
  ActivityIndicator,
  Animated
} from 'react-native';

import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import { useState, useRef } from 'react';
import axios from 'axios';
import { SafeAreaView } from 'react-native-safe-area-context';

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();
  const [photo, setPhoto] = useState(null);
  const [placa, setPlaca] = useState("");
  const [boxes, setBoxes] = useState([]);
  const [historial, setHistorial] = useState([]);
  const [loading, setLoading] = useState(false);

  const scaleAnim = useRef(new Animated.Value(1)).current;
  const cameraRef = useRef(null);

  const API_BASE = "http://107.21.34.72";

  const API_LARAVEL = `${API_BASE}/api/detectar-placa`;
  const API_YOLO = `${API_BASE}:8080/predict/`;

  if (!permission) return <View />;

  if (!permission.granted) {
    return (
      <SafeAreaView style={styles.center}>
        <Text style={styles.text}>Necesitamos permisos de cámara</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Permitir</Text>
        </TouchableOpacity>
      </SafeAreaView>
    );
  }

  const enviarImagen = async (uri) => {
    const formData = new FormData();

    formData.append("file", {
      uri,
      name: "photo.jpg",
      type: "image/jpeg",
    });

    try {
      setLoading(true);
      setPlaca("");
      setBoxes([]);

      // 🔹 1. DETECCIÓN IA
      const res = await axios.post(API_YOLO, formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const placaDetectada = res.data.placas?.[0] || "No detectada";

      setPlaca(placaDetectada);
      setBoxes(res.data.boxes || []);

      // 🔥 SI DETECTA PLACA REAL
      if (res.data.placas?.length) {

        // 🔹 2. ENVIAR A SMARTPARK (LARAVEL)
        await axios.post(API_LARAVEL, {
          placa: placaDetectada,
        });

        // 🔥 ANIMACIÓN
        Animated.sequence([
          Animated.timing(scaleAnim, {
            toValue: 1.2,
            duration: 150,
            useNativeDriver: true,
          }),
          Animated.timing(scaleAnim, {
            toValue: 1,
            duration: 150,
            useNativeDriver: true,
          }),
        ]).start();

        // 🔥 HISTORIAL
        setHistorial(prev => [placaDetectada, ...prev.slice(0, 4)]);
      }

    } catch (error) {
      console.log(error);
      setPlaca("Error de conexión");
    } finally {
      setLoading(false);
    }
  };

  const takePhoto = async () => {
    const result = await cameraRef.current.takePictureAsync();
    setPhoto(result.uri);
    enviarImagen(result.uri);
  };

  const pickImage = async () => {
    let result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 0.8,
    });

    if (!result.canceled) {
      const uri = result.assets[0].uri;
      setPhoto(uri);
      enviarImagen(uri);
    }
  };

  const reset = () => {
    setPhoto(null);
    setPlaca("");
    setBoxes([]);
  };

  return (
    <SafeAreaView style={styles.container}>

      {/* HEADER */}
      <View style={styles.header}>
        <Text style={styles.title}>🚗 SmartPark IA</Text>
        <Text style={styles.subtitle}>Detección automática de placas</Text>
      </View>

      {/* PREVIEW */}
      <View style={styles.preview}>
        {!photo ? (
          <CameraView style={styles.camera} ref={cameraRef} />
        ) : (
          <Image source={{ uri: photo }} style={styles.image} />
        )}

        {/* 🔥 BOUNDING BOX */}
        {boxes.map((box, index) => {
          const [x1, y1, x2, y2] = box;

          return (
            <View
              key={index}
              style={{
                position: "absolute",
                left: x1,
                top: y1,
                width: x2 - x1,
                height: y2 - y1,
                borderWidth: 2,
                borderColor: "#22c55e",
                borderRadius: 5,
              }}
            />
          );
        })}

        {loading && (
          <View style={styles.loader}>
            <ActivityIndicator size="large" color="#22c55e" />
            <Text style={styles.loadingText}>Analizando...</Text>
          </View>
        )}
      </View>

      {/* BOTONES */}
      <View style={styles.buttons}>
        <TouchableOpacity style={styles.button} onPress={takePhoto}>
          <Text style={styles.buttonText}>📸 Tomar</Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.buttonSecondary} onPress={pickImage}>
          <Text style={styles.buttonText}>🖼️ Galería</Text>
        </TouchableOpacity>

        {photo && (
          <TouchableOpacity style={styles.buttonReset} onPress={reset}>
            <Text style={styles.buttonText}>♻️ Reset</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* RESULTADO */}
      <View style={styles.result}>
        <Text style={styles.resultText}>Placa detectada</Text>

        <Animated.Text
          style={[styles.placa, { transform: [{ scale: scaleAnim }] }]}
        >
          {loading ? "..." : placa || "---"}
        </Animated.Text>
      </View>

      {/* HISTORIAL */}
      <View style={{ marginBottom: 10 }}>
        <Text style={{ color: "white", marginBottom: 5 }}>Historial:</Text>
        {historial.map((p, i) => (
          <Text key={i} style={{ color: "#94a3b8" }}>
            • {p}
          </Text>
        ))}
      </View>

    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0f172a",
    paddingHorizontal: 15,
  },
  header: {
    alignItems: "center",
    marginTop: 20,
    marginBottom: 10,
  },
  title: {
    fontSize: 24,
    color: "white",
    fontWeight: "bold",
  },
  subtitle: {
    color: "#94a3b8",
    fontSize: 12,
  },
  preview: {
    flex: 1,
    borderRadius: 15,
    overflow: "hidden",
    marginBottom: 15,
    position: "relative",
  },
  camera: { flex: 1 },
  image: { flex: 1, resizeMode: "cover" },
  loader: {
    position: "absolute",
    top: "40%",
    alignSelf: "center",
    alignItems: "center",
  },
  loadingText: {
    color: "white",
    marginTop: 5,
  },
  buttons: {
    flexDirection: "row",
    gap: 10,
    marginBottom: 15,
  },
  button: {
    flex: 1,
    backgroundColor: "#22c55e",
    padding: 12,
    borderRadius: 10,
    alignItems: "center",
  },
  buttonSecondary: {
    flex: 1,
    backgroundColor: "#3b82f6",
    padding: 12,
    borderRadius: 10,
    alignItems: "center",
  },
  buttonReset: {
    flex: 1,
    backgroundColor: "#ef4444",
    padding: 12,
    borderRadius: 10,
    alignItems: "center",
  },
  buttonText: {
    color: "white",
    fontWeight: "bold",
  },
  result: {
    backgroundColor: "#1e293b",
    padding: 15,
    borderRadius: 10,
    alignItems: "center",
    marginBottom: 10,
  },
  resultText: {
    color: "#94a3b8",
  },
  placa: {
    color: "#22c55e",
    fontSize: 24,
    fontWeight: "bold",
    marginTop: 5,
  },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  text: {
    color: "white",
  },
});