@main def exec(inputPath) = ={
    importCode(inputPath)
    java.nio.file.Files.createDirectories(
        java.nio.file.Paths.get("/home/demo/outputs")
    )
    cpg.method.filterNot(_.isExternal).foreach { method => 
        cpg.method.id(method.id).dotPdg #> s"/home/demo/outputs/${method.id}.dot"
    }
}