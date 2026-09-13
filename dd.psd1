@{
    schema = 1
    project = @{
        name = 'vector-experiments'
        type = 'cli'
        'default-target' = 'app'
    }
    dependencies = @{ owner = 'dd' }
    build = @{
        'x64-windows' = @{
            debug = 'debug'
            release = 'release'
        }
        'x64-linux' = @{
            debug = 'debug'
            release = 'release'
        }
    }
    targets = @(
        @{
            id = 'app'
            kind = 'cli'
            'cmake-target' = 'vector-experiments'
            'test-label' = 'vector-experiments'
            'debug-path' = 'build/debug/vector-experiments{exe}'
            'release-path' = 'build/release/vector-experiments{exe}'
        }
    )
}
