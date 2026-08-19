//! Default memory-intent routes from Meliclaw Routing Inteligente (Capa 1).
//! These classify query *type*; they do not query stores (Capa 2 lives elsewhere).

use crate::route::Route;

pub fn memory_intent_routes() -> Vec<Route> {
    vec![
        Route::new(
            "factual",
            vec![
                "NIF de",
                "dirección de",
                "teléfono de",
                "fecha de nacimiento",
                "cuál es el NIF del cliente",
            ],
        ),
        Route::new(
            "temporal",
            vec![
                "cuándo cambió",
                "historia de",
                "evolución de",
                "antes era",
                "qué decidí ayer sobre",
            ],
        ),
        Route::new(
            "preference",
            vec!["prefiero", "me gusta", "tono preferido", "estilo"],
        ),
        Route::new(
            "procedural",
            vec!["cómo resolví", "pasos para", "procedimiento de"],
        ),
        Route::new(
            "relational",
            vec!["quién trabaja con", "reporta a", "pertenece a", "equipo de"],
        ),
        Route::new(
            "semantic",
            vec!["todas las personas que", "empresas del sector", "inferir"],
        ),
        Route::new(
            "keyword_search",
            vec!["busca", "encuentra el texto", "refactorización del módulo"],
        ),
        Route::new(
            "vector_similarity",
            vec![
                "concepto similar",
                "vector search por",
                "semánticamente parecido",
            ],
        ),
    ]
}
