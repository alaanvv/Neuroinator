#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define PRINT(...) { printf(__VA_ARGS__); printf("\n"); }

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef float    f32;
typedef double   f64;

// ---

typedef enum { INN, HID, OUT } LayerType;

typedef struct {
  f32 inn, bias;
  f32* weights;
} Node;

struct Layer;
typedef struct {
  LayerType type;
  u8 size;
  Node* nodes;
  struct Layer* next;
} Layer;

typedef struct {
  u8 size;
  Layer* layers;
  u8 num_inns, num_outs;
  f32* inns, outs;
} Network;

// ---

Network* create_neural_network(u8 num_inn, u8 num_out) {
  Network* network = malloc(sizeof(Network));
  network->size = 2;
  network->num_inns = num_inn;
  network->num_outs = num_out;

  Node* inn_nodes = malloc(num_inn * sizeof(Node));
  Node* out_nodes = malloc(num_out * sizeof(Node));

  Layer* layers = malloc(2 * sizeof(Layer));

  layers[0].type = INN;
  layers[0].size = num_inn;
  layers[0].nodes = inn_nodes;
  for (u8 i = 0; i < layers[0].size; i++) 
    layers[0].nodes[i].weights = malloc(layers[1].size * sizeof(f32));
  layers[0].next = (struct Layer*) &layers[1];

  layers[1].type = OUT;
  layers[1].size = num_out;
  layers[1].nodes = out_nodes;
  layers[1].next = (struct Layer*) NULL;

  network->layers = layers;

  return network;
}

void randomize_nodes(Layer layer, f32 min_w, f32 max_w, f32 min_b, f32 max_b) {
  if (layer.type != OUT) randomize_nodes(*(Layer*) layer.next, min_w, max_w, min_b, max_b);

  for (u8 i = 0; i < layer.size; i++) {
    layer.nodes[i].bias = fmod(random() / 1e5, (max_b - min_b)) + min_b;

    if (layer.type == OUT) continue;

    for (u8 j = 0; j < ((Layer*) layer.next)->size; j++) {
      layer.nodes[i].weights[j] = fmod(random() / 1e5, (max_w - min_w)) + min_w;
    }
  }
}

// ---

u8 main() {
  srand(time(0));

  Network* network = create_neural_network(2, 1);
  randomize_nodes(network->layers[0], 0, 1, 0, 1);

  PRINT("%f", network->layers[0].nodes[0].weights[0]);

  return 0;
}
