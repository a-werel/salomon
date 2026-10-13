@main def exec(inputPath: String) = {
    importCode(inputPath)
    run.ossdataflow

    java.nio.file.Files.createDirectories(
        java.nio.file.Paths.get("/home/demo/output")
    )
    cpg.method.filterNot(_.isExternal).foreach { method => 
        cpg.method.id(method.id).dotPdg #> s"/home/demo/output/${method.id}.dot"
    }
}