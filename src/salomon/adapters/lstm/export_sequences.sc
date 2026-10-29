import io.shiftleft.codepropertygraph.generated.nodes

@main def exec(inputPath: String, outputPath: String) = {
    importCode(inputPath)

    def preorder(node: nodes.AstNode): List[String] = {
        val token = node match {
            case call: nodes.Call       => s"CALL:${call.name}"
            case literal: nodes.Literal => s"LITERAL:${literal.code}"
            case control: nodes.ControlStructure => s"CONTROL:${control.controlStructureType}"
            case other => other.label
        }
        val children = Iterator.single(node).astChildren.l.sortBy(_.order)
        List(token) ++ children.flatMap(preorder) ++ List("END")
    }

    val methods = cpg.method.filterNot(_.isExternal)
        .filterNot(_.name.startsWith("<")).l.sortBy(m => (m.filename, m.fullName))
    val records = methods.map { method =>
        ujson.Obj(
            "source" -> method.filename,
            "function" -> method.name,
            "tokens" -> ujson.Arr.from(preorder(method))
        )
    }
    val path = java.nio.file.Paths.get(outputPath)
    java.nio.file.Files.createDirectories(path.getParent)
    java.nio.file.Files.writeString(path, ujson.write(ujson.Arr.from(records), indent = 2))
}
